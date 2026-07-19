"""Workflow-Progress and Residency-Aware A2C (WPR-A2C).

This version fixes the earlier critical issue where the actor only saw a
state-level scalar shared by all actions. The actor now scores each candidate
assignment with an action feature vector phi(S, a), and the policy-gradient
update uses the correct softmax-linear gradient:

    grad log pi(a|S) = phi(S,a) - sum_a' pi(a'|S) phi(S,a').

代码中保留中文注释，方便后续整理到论文和组会材料。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from dag_a2c.wpr_env import STAGE_TYPES, WPREnv


WAIT_SLOT = -1
WAIT_STAGE = -1
WAIT_MODEL = -1


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    z -= float(np.max(z))
    p = np.exp(z)
    return p / max(float(np.sum(p)), 1e-12)


@dataclass
class WPRA2CConfig:
    actor_lr: float = 0.004
    critic_lr: float = 0.010
    demand_lr: float = 0.006
    entropy_coef: float = 0.001
    beta_time_discount: float = 0.035
    prep_lambda: float = 0.38
    wait_bias: float = -0.08
    temperature: float = 0.90
    seed: int = 0
    use_progress_encoder: bool = True
    use_demand_predictor: bool = True
    use_residency_scorer: bool = True
    use_residency_features: bool = True
    use_wait_features: bool = True
    use_time_critic: bool = True
    allow_wait: bool = True
    use_potential_shaping: bool = True
    gae_lambda: float = 0.92
    advantage_clip: float = 4.0
    return_best_checkpoint: bool = True
    checkpoint_metric: str = "weighted_completed_value"
    structural_prior_strength: float = 1.5
    progress_action_scale: float = 1.0
    validation_interval: int = 0
    validation_episodes: int = 1
    validation_seed_offset: int = 900000


class WorkflowProgressEncoder:
    """模块 1：Permutation-invariant workflow/GPU progress encoder.

    旧实现直接 flatten active workflow slots。workflow 完成后 slot 会移动，导致同一
    位置语义不稳定。这里改为 DeepSets 风格的池化表示：
    - workflow feature mean / max / normalized sum；
    - GPU residency feature mean / max；
    - global event state。
    """

    def encode(self, env: WPREnv, use_progress: bool = True) -> np.ndarray:
        obs = env.observe()
        wf = obs["workflow_features"].copy()
        if not use_progress:
            wf[:] = 0.0
        active = wf[:, 0] > 0.0
        if np.any(active):
            wf_active = wf[active]
            wf_mean = np.mean(wf_active, axis=0)
            wf_max = np.max(wf_active, axis=0)
            wf_sum = np.sum(wf_active, axis=0) / max(1.0, float(env.max_active))
        else:
            wf_mean = np.zeros(wf.shape[1], dtype=np.float32)
            wf_max = np.zeros(wf.shape[1], dtype=np.float32)
            wf_sum = np.zeros(wf.shape[1], dtype=np.float32)

        gpu = obs["residency"]
        gpu_mean = np.mean(gpu, axis=0)
        gpu_max = np.max(gpu, axis=0)
        count = np.asarray([np.sum(active) / max(1, env.max_active)], dtype=np.float32)
        return np.concatenate([wf_mean, wf_max, wf_sum, gpu_mean, gpu_max, obs["global"], count]).astype(np.float32)


class FutureModelDemandPredictor:
    """模块 2：Future Model-Demand Predictor.

    为避免“预测函数与梯度不匹配”，这里使用线性输出 + MSE，梯度严格对应
    pred = state @ W + b。用于 scorer 时再做 clip，训练本身不经过 softplus/max norm。
    """

    def __init__(self, state_dim: int, num_models: int, rng: np.random.Generator) -> None:
        self.W = rng.normal(0.0, 0.015, size=(state_dim, num_models)).astype(np.float32)
        self.b = np.zeros(num_models, dtype=np.float32)

    def predict_raw(self, state: np.ndarray) -> np.ndarray:
        return (state @ self.W + self.b).astype(np.float32)

    def predict(self, state: np.ndarray) -> np.ndarray:
        return np.clip(self.predict_raw(state), 0.0, 1.5)

    def update(self, state: np.ndarray, target: np.ndarray, lr: float) -> float:
        pred = self.predict_raw(state)
        err = pred - target
        self.W -= lr * np.outer(state, err)
        self.b -= lr * err
        return float(np.mean(err**2))


class TimeAwareCritic:
    """模块 5：Time-aware critic with pooled state features."""

    def __init__(self, state_dim: int, rng: np.random.Generator, hidden_dim: int = 64) -> None:
        self.W1 = rng.normal(0.0, 0.06, size=(state_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0.0, 0.03, size=hidden_dim).astype(np.float32)
        self.b2 = 0.0

    def value(self, state: np.ndarray) -> float:
        h = np.maximum(state @ self.W1 + self.b1, 0.0)
        return float(h @ self.W2 + self.b2)

    def update(self, state: np.ndarray, target: float, lr: float) -> float:
        h_pre = state @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)
        value = float(h @ self.W2 + self.b2)
        advantage = float(np.clip(target - value, -10.0, 10.0))
        old_w2 = self.W2.copy()
        self.W2 += lr * advantage * h
        self.b2 += lr * advantage
        dh = advantage * old_w2 * (h_pre > 0.0)
        self.W1 += lr * np.outer(state, dh)
        self.b1 += lr * dh
        return advantage


class WPRA2CAgent:
    """WPR-A2C 主体。

    Actor 参数作用在 action feature 上，而不是只作用在 state 上。
    """

    def __init__(self, env: WPREnv, config: WPRA2CConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.encoder = WorkflowProgressEncoder()
        state_dim = len(self.encode(env))
        self.demand = FutureModelDemandPredictor(state_dim, env.num_models, self.rng)
        self.critic = TimeAwareCritic(state_dim, self.rng)
        self.action_dim = len(self.action_features(env, self.encode(env), np.zeros(env.num_models, dtype=np.float32), (WAIT_SLOT, WAIT_STAGE, WAIT_MODEL, 0)))
        self.actor_w = self.rng.normal(0.0, 0.025, size=self.action_dim).astype(np.float32)
        self.actor_b = 0.0
        self.apply_structural_actor_prior(env)

    def apply_structural_actor_prior(self, env: WPREnv) -> None:
        """Initialize the actor with the problem-structured cross scorer prior.

        这不是固定启发式策略；它只是 actor 的初始参数。后续 A2C 更新仍会根据
        rollout advantage 改写这些权重。这样可以降低随机初始化在组合动作空间中的
        探索成本，并让论文中的 residency-aware scorer 真正进入策略参数。
        """

        strength = float(self.config.structural_prior_strength)
        if strength <= 0.0:
            return
        wf_dim = env.workflow_progress_features().shape[1]
        stage_offset = wf_dim + env.num_stage_types
        model_offset = stage_offset + 5
        cross_offset = self.action_dim - 19
        prior = np.zeros_like(self.actor_w)

        # stage_scalar: work, quality requirement, slack ratio, remaining critical path, service weight.
        prior[stage_offset + 0] -= 0.20
        prior[stage_offset + 2] -= 0.18
        prior[stage_offset + 3] += 0.28
        prior[stage_offset + 4] += 0.55

        # model_scalar: quality, prefill, decode, memory, weight size, adapter size, feasible, predicted demand.
        prior[model_offset + 0] += 0.55
        prior[model_offset + 1] -= 0.25
        prior[model_offset + 2] -= 0.30
        prior[model_offset + 3] -= 0.12
        prior[model_offset + 4] -= 0.12
        prior[model_offset + 5] -= 0.05
        prior[model_offset + 6] += 0.35
        prior[model_offset + 7] += 0.28

        # cross features: prep, communication, execution, residency hit/backbone, demand and priority terms.
        prior[cross_offset + 0] -= 0.50
        prior[cross_offset + 1] -= 0.28
        prior[cross_offset + 2] -= 0.55
        prior[cross_offset + 3] += 0.80
        prior[cross_offset + 4] += 0.22
        prior[cross_offset + 5] += 0.30
        prior[cross_offset + 6] -= 0.12
        prior[cross_offset + 7] += 0.36
        prior[cross_offset + 8] += 0.50
        prior[cross_offset + 9] += 1.00
        prior[cross_offset + 10] += 0.18
        prior[cross_offset + 14] += 0.18
        prior[cross_offset + 18] += 0.05
        self.actor_w += strength * prior.astype(np.float32)

    def encode(self, env: WPREnv) -> np.ndarray:
        return self.encoder.encode(env, use_progress=self.config.use_progress_encoder)

    def predict_demand(self, state: np.ndarray, env: WPREnv) -> np.ndarray:
        if self.config.use_demand_predictor:
            return self.demand.predict(state)
        return np.zeros(env.num_models, dtype=np.float32)

    def dispatch(self, env: WPREnv, deterministic: bool = False) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
        """模块 4：Event-aware autoregressive matching decoder.

        对每个 idle GPU 按固定顺序选择动作。候选集合现在包含 WAIT_g：
        - dispatch action: (workflow_slot, stage_id, model_id, gpu_id)
        - wait action:     (-1, -1, -1, gpu_id)
        """

        state = self.encode(env)
        demand = self.predict_demand(state, env)
        demand_target = env.oracle_dag_demand_target()
        assignments: list[tuple[int, int, int, int]] = []
        records: list[dict] = []
        used_pairs: set[tuple[int, int]] = set()

        idle_gpus = sorted(env.idle_gpus())
        if self.config.allow_wait and idle_gpus and env.has_future_external_event() and env.ready_pairs():
            # 全局 WAIT_ALL：只有当策略决定本轮完全不调度时才等待，避免局部 WAIT
            # 造成同一 timestamp 反复重新决策。
            wait_action = (WAIT_SLOT, WAIT_STAGE, WAIT_MODEL, -1)
            wait_candidates = [wait_action]
            for candidate_gpu in idle_gpus:
                wait_candidates.extend(env.feasible_actions_for_gpu(candidate_gpu, used_pairs))
            features = np.asarray([self.action_features(env, state, demand, a) for a in wait_candidates], dtype=np.float32)
            logits = features @ self.actor_w + self.actor_b
            logits = logits + np.asarray([self.config.wait_bias if a[0] < 0 else 0.0 for a in wait_candidates], dtype=np.float32)
            probs = softmax(logits, self.config.temperature)
            idx = int(np.argmax(logits)) if deterministic else int(self.rng.choice(len(wait_candidates), p=probs))
            records.append(
                {
                    "state": state,
                    "demand_target": demand_target,
                    "action": wait_candidates[idx],
                    "candidates": wait_candidates,
                    "features": features,
                    "probs": probs,
                    "logits": logits,
                }
            )
            if wait_candidates[idx][0] < 0:
                return [wait_action], records
            assignments.append(wait_candidates[idx])
            used_pairs.add((wait_candidates[idx][0], wait_candidates[idx][1]))

        for gpu_id in idle_gpus:
            if any(a[3] == gpu_id for a in assignments):
                continue
            candidates = env.feasible_actions_for_gpu(gpu_id, used_pairs)
            if not candidates:
                continue
            features = np.asarray([self.action_features(env, state, demand, a) for a in candidates], dtype=np.float32)
            logits = features @ self.actor_w + self.actor_b
            probs = softmax(logits, self.config.temperature)
            idx = int(np.argmax(logits)) if deterministic else int(self.rng.choice(len(candidates), p=probs))
            action = candidates[idx]
            records.append(
                {
                    "state": state,
                    "demand_target": demand_target,
                    "action": action,
                    "candidates": candidates,
                    "features": features,
                    "probs": probs,
                    "logits": logits,
                }
            )
            if action[0] >= 0:
                assignments.append(action)
                used_pairs.add((action[0], action[1]))
            else:
                assignments.append(action)
        return assignments, records

    def action_features(self, env: WPREnv, state: np.ndarray, demand: np.ndarray, action: tuple[int, int, int, int]) -> np.ndarray:
        """构造 phi(S,a)：每个候选动作都有不同特征，actor 才能学习相对偏好。"""

        slot, stage_id, model_id, gpu_id = action
        wf_dim = env.workflow_progress_features().shape[1]
        wf_feat = np.zeros(wf_dim, dtype=np.float32)
        stage_type_onehot = np.zeros(env.num_stage_types, dtype=np.float32)
        stage_scalar = np.zeros(5, dtype=np.float32)
        model_scalar = np.zeros(8, dtype=np.float32)
        if slot < 0:
            gpu_scalar = self.global_wait_gpu_features(env)
            if not self.config.use_wait_features:
                gpu_scalar[:] = 0.0
        else:
            gpu = env.gpus[gpu_id]
            gpu_scalar = np.asarray(
                [
                    gpu.speed / max(g.speed for g in env.gpus),
                    gpu.memory / max(g.memory for g in env.gpus),
                    gpu.bandwidth / max(g.bandwidth for g in env.gpus),
                    float(env.resident_model[gpu_id] >= 0),
                ],
                dtype=np.float32,
            )
        cross = np.zeros(19, dtype=np.float32)
        is_wait = float(slot < 0)

        if slot >= 0:
            wf = env.active[slot]
            stage = wf.template.stages[stage_id]
            model = env.models[model_id]
            if self.config.use_progress_encoder:
                wf_feat = env.workflow_progress_features()[slot] * self.config.progress_action_scale
            stage_type_onehot[stage.stage_type] = 1.0
            slack = wf.arrival + wf.template.deadline - env.time
            remaining_cp = env.remaining_critical_path(wf)
            prep = env.prep_time(model_id, gpu_id)
            input_delay = env.input_transfer_delay(wf, stage_id, gpu_id)
            exec_t = env.expected_exec_time(slot, stage_id, model_id, gpu_id)
            current = int(env.resident_model[gpu_id])
            resident_hit = float(current == model_id)
            same_backbone = float(current >= 0 and env.models[current].backbone == model.backbone)
            current_demand = demand[current] if current >= 0 else 0.0
            delta_psi = self.residency_delta(env, demand, action) if self.config.use_residency_scorer else 0.0
            if not self.config.use_residency_features:
                resident_hit = 0.0
                same_backbone = 0.0
                current_demand = 0.0
                delta_psi = 0.0
            best_immediate = self.best_immediate_score(env)
            next_arrival, next_completion = self.next_event_features(env)
            stage_scalar = np.asarray(
                [
                    stage.work / 8.0,
                    stage.min_quality,
                    slack / max(wf.template.deadline, 1e-6),
                    remaining_cp / max(wf.template.deadline, 1e-6),
                    wf.template.weight / 5.0,
                ],
                dtype=np.float32,
            )
            model_scalar = np.asarray(
                [
                    model.quality_by_type[stage.stage_type],
                    model.prefill_time_per_ktok,
                    model.decode_time_per_ktok,
                    model.memory / max(m.memory for m in env.models),
                    model.weight_size / max(m.weight_size for m in env.models),
                    model.adapter_size / max(m.adapter_size for m in env.models),
                    float(stage.stage_type in model.supported_types),
                    demand[model_id],
                ],
                dtype=np.float32,
            )
            cross = np.asarray(
                [
                    prep / 4.0,
                    input_delay / 4.0,
                    exec_t / 10.0,
                    resident_hit,
                    same_backbone,
                    demand[model_id],
                    current_demand,
                    demand[model_id] - current_demand,
                    delta_psi,
                    wf.template.weight / max(0.5, slack),
                    float(len(env.ready_stages(wf))) / max(1, len(wf.template.stages)),
                    next_arrival,
                    next_completion,
                    self.min_ready_slack(env),
                    best_immediate,
                    len(env.ready_pairs()) / max(1, env.max_active * env.max_stages),
                    len(env.idle_gpus()) / max(1, env.num_gpus),
                    is_wait,
                    1.0,
                ],
                dtype=np.float32,
            )
        else:
            resident_demands = [float(demand[int(m)]) for m in env.resident_model if int(m) >= 0]
            max_resident_demand = max(resident_demands, default=0.0)
            mean_resident_demand = float(np.mean(resident_demands)) if resident_demands else 0.0
            if not self.config.use_wait_features:
                max_resident_demand = 0.0
                mean_resident_demand = 0.0
            next_arrival, next_completion = self.next_event_features(env)
            wait_next_arrival = next_arrival if self.config.use_wait_features else 0.0
            wait_next_completion = next_completion if self.config.use_wait_features else 0.0
            wait_min_slack = self.min_ready_slack(env) if self.config.use_wait_features else 0.0
            wait_best_immediate = self.best_immediate_score(env) if self.config.use_wait_features else 0.0
            wait_ready_ratio = len(env.ready_pairs()) / max(1, env.max_active * env.max_stages) if self.config.use_wait_features else 0.0
            wait_idle_ratio = len(env.idle_gpus()) / max(1, env.num_gpus) if self.config.use_wait_features else 0.0
            cross = np.asarray(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    mean_resident_demand,
                    max_resident_demand,
                    mean_resident_demand - max_resident_demand,
                    0.0,
                    0.0,
                    wait_next_arrival,
                    wait_next_completion,
                    wait_min_slack,
                    wait_best_immediate,
                    wait_ready_ratio,
                    wait_idle_ratio,
                    is_wait,
                    1.0,
                ],
                dtype=np.float32,
            )

        return np.concatenate([wf_feat, stage_type_onehot, stage_scalar, model_scalar, gpu_scalar, cross]).astype(np.float32)

    def global_wait_gpu_features(self, env: WPREnv) -> np.ndarray:
        """Global GPU aggregation for WAIT_ALL rather than binding WAIT to one GPU."""

        idle = env.idle_gpus()
        gpus = [env.gpus[g] for g in idle] if idle else list(env.gpus)
        max_speed = max(g.speed for g in env.gpus)
        max_mem = max(g.memory for g in env.gpus)
        max_bw = max(g.bandwidth for g in env.gpus)
        resident_present = [float(env.resident_model[g.gpu_id] >= 0) for g in gpus]
        return np.asarray(
            [
                float(np.mean([g.speed / max_speed for g in gpus])),
                float(np.mean([g.memory / max_mem for g in gpus])),
                float(np.mean([g.bandwidth / max_bw for g in gpus])),
                float(np.mean(resident_present)) if resident_present else 0.0,
            ],
            dtype=np.float32,
        )

    def residency_delta(self, env: WPREnv, demand: np.ndarray, action: tuple[int, int, int, int]) -> float:
        slot, stage_id, model_id, gpu_id = action
        if slot < 0:
            # WAIT 保留当前 resident model；如果当前模型未来需求高，则等待更有价值。
            current = int(env.resident_model[gpu_id])
            return float(0.35 * (demand[current] if current >= 0 else 0.0))
        replaced = int(env.resident_model[gpu_id])
        replaced_demand = demand[replaced] if replaced >= 0 else 0.0
        return float(demand[model_id] - replaced_demand - self.config.prep_lambda * env.prep_time(model_id, gpu_id))

    def next_event_features(self, env: WPREnv) -> tuple[float, float]:
        next_arrival = np.inf
        if env.pending:
            next_arrival = max(0.0, float(env.pending[0].arrival - env.time))
        next_completion = np.inf
        if env.running:
            next_completion = min(max(0.0, float(x["finish"]) - env.time) for x in env.running)
        scale = max(1.0, env.horizon)
        return (
            float(min(next_arrival, scale) / scale),
            float(min(next_completion, scale) / scale),
        )

    def min_ready_slack(self, env: WPREnv) -> float:
        vals = []
        for slot, _sid in env.ready_pairs():
            wf = env.active[slot]
            vals.append((wf.arrival + wf.template.deadline - env.time) / max(wf.template.deadline, 1e-6))
        return float(np.clip(min(vals, default=1.0), -1.0, 1.0))

    def best_immediate_score(self, env: WPREnv) -> float:
        best = -10.0
        for g in env.idle_gpus():
            for slot, sid, mid, gid in env.feasible_actions_for_gpu(g):
                wf = env.active[slot]
                slack = wf.arrival + wf.template.deadline - env.time
                score = wf.template.weight / max(0.5, slack) - 0.20 * (env.input_transfer_delay(wf, sid, gid) + env.prep_time(mid, gid) + env.expected_exec_time(slot, sid, mid, gid))
                best = max(best, float(score))
        return float(np.clip(best, -10.0, 10.0) / 10.0)

    def update_policy_heads(self, records: list[dict], advantage: float) -> dict[str, float]:
        """Update actor and demand head with a precomputed advantage."""

        if not records:
            return {"advantage": 0.0, "demand_loss": 0.0, "entropy": 0.0}
        state = records[0]["state"]
        policy_advantage = float(np.clip(advantage, -self.config.advantage_clip, self.config.advantage_clip))
        entropies = []
        for rec in records:
            features = rec["features"]
            probs = rec["probs"]
            chosen_idx = rec["candidates"].index(rec["action"])
            expected_feature = probs @ features
            grad_logp = (features[chosen_idx] - expected_feature) / max(self.config.temperature, 1e-6)

            # Entropy gradient for a softmax-linear policy.
            entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-9))))
            entropy_logit_grad = -probs * (np.log(np.maximum(probs, 1e-9)) + entropy)
            entropy_feature_grad = (entropy_logit_grad @ features) / max(self.config.temperature, 1e-6)

            self.actor_w += self.config.actor_lr * (policy_advantage * grad_logp + self.config.entropy_coef * entropy_feature_grad)
            entropies.append(entropy)

        demand_loss = 0.0
        if self.config.use_demand_predictor:
            demand_loss = self.demand.update(state, records[0]["demand_target"], self.config.demand_lr)
        return {"advantage": float(advantage), "demand_loss": float(demand_loss), "entropy": float(np.mean(entropies)) if entropies else 0.0}

    def update(self, records: list[dict], reward: float, next_state: np.ndarray, done: bool, dt: float) -> dict[str, float]:
        """Backward-compatible one-step update."""

        if not records:
            return {"advantage": 0.0, "demand_loss": 0.0, "entropy": 0.0}
        state = records[0]["state"]
        discount = np.exp(-self.config.beta_time_discount * dt) if self.config.use_time_critic else 0.97
        target = reward if done else reward + discount * self.critic.value(next_state)
        advantage = self.critic.update(state, target, self.config.critic_lr)
        return self.update_policy_heads(records, advantage)

    def update_from_gae(self, transitions: list[dict]) -> dict[str, float]:
        """Event-aware GAE update over one rollout episode."""

        indexed = [tr for tr in transitions if tr["records"]]
        if not indexed:
            return {"advantage": 0.0, "mean_abs_advantage": 0.0, "demand_loss": 0.0, "entropy": 0.0}

        states = [tr["records"][0]["state"] for tr in indexed]
        values = np.asarray([self.critic.value(s) for s in states], dtype=np.float32)
        next_values = np.asarray([0.0 if tr["done"] else self.critic.value(tr["next_state"]) for tr in indexed], dtype=np.float32)
        gammas = np.asarray(
            [np.exp(-self.config.beta_time_discount * tr["dt"]) if self.config.use_time_critic else 0.97 for tr in indexed],
            dtype=np.float32,
        )
        rewards = np.asarray([tr["reward"] for tr in indexed], dtype=np.float32)

        advantages = np.zeros(len(indexed), dtype=np.float32)
        gae = 0.0
        for idx in range(len(indexed) - 1, -1, -1):
            delta = rewards[idx] + gammas[idx] * next_values[idx] - values[idx]
            gae = float(delta + gammas[idx] * self.config.gae_lambda * gae)
            advantages[idx] = gae
        returns = values + advantages

        out = {"advantage": [], "mean_abs_advantage": [], "demand_loss": [], "entropy": []}
        for tr, state, advantage, target in zip(indexed, states, advantages, returns):
            self.critic.update(state, float(target), self.config.critic_lr)
            stats = self.update_policy_heads(tr["records"], float(advantage))
            out["advantage"].append(float(advantage))
            out["mean_abs_advantage"].append(abs(float(advantage)))
            out["demand_loss"].append(stats["demand_loss"])
            out["entropy"].append(stats["entropy"])
        return {key: float(np.mean(vals)) if vals else 0.0 for key, vals in out.items()}


def train_wpr_agent(env_factory, episodes: int, seed: int, config: WPRA2CConfig | None = None) -> tuple[WPRA2CAgent, list[dict[str, float]]]:
    probe = env_factory(seed)
    cfg = config or WPRA2CConfig(seed=seed)
    probe.enable_potential_shaping = cfg.use_potential_shaping
    probe.shaping_beta = cfg.beta_time_discount
    agent = WPRA2CAgent(probe, cfg)
    curve: list[dict[str, float]] = []
    best_agent = copy.deepcopy(agent)
    best_score = -float("inf")

    def validation_score(candidate: WPRA2CAgent, ep: int) -> float:
        scores = []
        for vidx in range(max(1, cfg.validation_episodes)):
            env = env_factory(seed + cfg.validation_seed_offset + 1000 * ep + vidx)
            env.enable_potential_shaping = cfg.use_potential_shaping
            env.shaping_beta = cfg.beta_time_discount
            env.reset(seed + cfg.validation_seed_offset + 1000 * ep + vidx)
            while not env.done:
                assignments, _ = candidate.dispatch(env, deterministic=True)
                env.step(assignments)
            metrics = env.final_metrics()
            scores.append(float(metrics.get(cfg.checkpoint_metric, metrics["weighted_completed_value"])))
        return float(np.mean(scores))

    for ep in range(episodes):
        env = env_factory(seed + ep)
        env.enable_potential_shaping = cfg.use_potential_shaping
        env.shaping_beta = cfg.beta_time_discount
        env.reset(seed + ep)
        transitions = []
        while not env.done:
            assignments, records = agent.dispatch(env, deterministic=False)
            _, reward, done, info = env.step(assignments)
            transitions.append(
                {
                    "records": records,
                    "reward": float(reward),
                    "next_state": agent.encode(env),
                    "done": bool(done),
                    "dt": float(info.get("dt", 0.0)),
                }
            )
        stats = agent.update_from_gae(transitions)
        metrics = env.final_metrics()
        curve.append(
            {
                "episode": float(ep),
                "weighted_completed_value": metrics["weighted_completed_value"],
                "weighted_goodput": metrics["weighted_goodput"],
                "weighted_goodput_rate": metrics["weighted_goodput_rate"],
                "sla_success_ratio": metrics["sla_success_ratio"],
                "p95_latency": metrics["p95_latency"],
                "demand_loss": stats["demand_loss"],
                "mean_abs_advantage": stats["mean_abs_advantage"],
                "policy_entropy": stats["entropy"],
            }
        )
        metric_score = float(metrics.get(cfg.checkpoint_metric, metrics["weighted_completed_value"]))
        if cfg.validation_interval > 0 and (ep + 1) % cfg.validation_interval == 0:
            metric_score = validation_score(agent, ep)
        if metric_score > best_score:
            best_score = metric_score
            best_agent = copy.deepcopy(agent)
    return (best_agent if cfg.return_best_checkpoint else agent), curve
