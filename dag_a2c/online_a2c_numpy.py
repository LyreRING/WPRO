"""Masked A2C for the online multi-request environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dag_a2c.online_env import OnlineDualDAGServingEnv


@dataclass
class OnlineA2CConfig:
    gamma: float = 0.97
    actor_lr: float = 0.00001
    critic_lr: float = 0.001
    entropy_coef: float = 0.002
    action_prior_coef: float = 4.0
    seed: int = 42


class OnlineMaskedLinearA2C:
    def __init__(self, state_dim: int, action_dims: tuple[int, int, int, int], config: OnlineA2CConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.actor_w = [self.rng.normal(0.0, 0.012, size=(state_dim, dim)).astype(np.float32) for dim in action_dims]
        self.actor_b = [np.zeros(dim, dtype=np.float32) for dim in action_dims]
        self.critic_w = np.zeros(state_dim, dtype=np.float32)
        self.critic_b = 0.0

    def act(self, state: np.ndarray, env: OnlineDualDAGServingEnv, deterministic: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
        masks = self._head_masks(env)
        self.last_masks = [mask.copy() for mask in masks]
        probs = [masked_softmax(state @ self.actor_w[idx] + self.actor_b[idx], masks[idx]) for idx in range(4)]
        action = self._select_joint_action(state, env, deterministic)
        return np.asarray(action, dtype=np.int64), probs

    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool, env: OnlineDualDAGServingEnv) -> float:
        value = self.value(state)
        target = reward if done else reward + self.config.gamma * self.value(next_state)
        advantage = float(np.clip(target - value, -25.0, 25.0))
        self.critic_w += self.config.critic_lr * advantage * state
        self.critic_b += self.config.critic_lr * advantage
        masks = getattr(self, "last_masks", self._head_masks(env))
        for head, selected in enumerate(action):
            logits = state @ self.actor_w[head] + self.actor_b[head]
            probs = masked_softmax(logits, masks[head])
            grad = -probs
            grad[int(selected)] += 1.0
            entropy_grad = -probs * (np.log(np.maximum(probs, 1e-8)) + 1.0)
            grad = advantage * grad + self.config.entropy_coef * entropy_grad
            self.actor_w[head] += self.config.actor_lr * np.outer(state, grad)
            self.actor_b[head] += self.config.actor_lr * grad
        return advantage

    def value(self, state: np.ndarray) -> float:
        return float(state @ self.critic_w + self.critic_b)

    def _head_masks(self, env: OnlineDualDAGServingEnv) -> list[np.ndarray]:
        masks = env.feasible_action_masks()
        return [masks["request"], masks["task"], masks["model"], masks["server"]]

    def _select_joint_action(self, state: np.ndarray, env: OnlineDualDAGServingEnv, deterministic: bool) -> tuple[int, int, int, int]:
        actions = env.feasible_actions()
        if not actions:
            return (0, 0, 0, 0)
        logits = [state @ self.actor_w[idx] + self.actor_b[idx] for idx in range(4)]
        scores = []
        for req_slot, task_id, model_id, server_id in actions:
            actor_score = logits[0][req_slot] + logits[1][task_id] + logits[2][model_id] + logits[3][server_id]
            score = 0.02 * actor_score
            score += self.config.action_prior_coef * self._action_prior(env, req_slot, task_id, model_id, server_id)
            scores.append(float(score))
        scores_arr = np.asarray(scores, dtype=np.float64)
        if deterministic:
            return actions[int(np.argmax(scores_arr))]
        probs = masked_softmax(scores_arr, np.ones(len(scores_arr), dtype=bool))
        return actions[int(self.rng.choice(len(actions), p=probs))]

    def _action_prior(self, env: OnlineDualDAGServingEnv, req_slot: int, task_id: int, model_id: int, server_id: int) -> float:
        req = env.active[req_slot]
        task = req.instance.request.tasks[task_id]
        model = env.models[model_id]
        duration = env.execution_duration(req, task_id, model_id, server_id)
        switch = env.transition_overhead(req, task_id, model_id, server_id)
        start = max(float(env.server_time[server_id]), env.current_time, req.arrival_time)
        finish = start + duration + switch
        deadline = req.arrival_time + req.instance.request.deadline
        projected = req.finish_times.copy()
        projected[task_id] = finish
        terminal = bool(np.all(req.scheduled | (np.arange(len(req.scheduled)) == task_id)))
        completion = float(np.nanmax(projected)) if terminal else finish + self._successor_work_lower_bound(env, req, task_id)
        violation = max(0.0, completion - deadline)
        cost = task.compute_demand * model.unit_cost + (duration + switch) * env.servers[server_id].operating_cost_per_time
        cashflow_signal = 0.0
        if terminal:
            cashflow_signal = req.instance.request.service_fee if violation <= 1e-6 else -req.instance.request.sla_penalty
        else:
            scale = max(0.15 * req.instance.request.deadline, 0.5)
            breach_risk = 1.0 / (1.0 + np.exp(-np.clip((completion - deadline) / scale, -20.0, 20.0)))
            remaining = max(1, int(np.sum(~req.scheduled)))
            cashflow_signal = ((1.0 - breach_risk) * req.instance.request.service_fee - breach_risk * req.instance.request.sla_penalty) / remaining
        return float((cashflow_signal - cost) / max(req.instance.request.service_fee, 1.0))

    def _successor_work_lower_bound(self, env: OnlineDualDAGServingEnv, req, task_id: int) -> float:
        graph = req.instance.request.task_graph
        memo: dict[int, float] = {}

        def min_service(node: int) -> float:
            candidates = []
            for model_id in np.where(req.instance.compatibility[node] > 0)[0]:
                for server_id in range(env.num_servers):
                    candidates.append(env.execution_duration(req, node, int(model_id), server_id))
            return min(candidates, default=0.0)

        def path(node: int) -> float:
            if node in memo:
                return memo[node]
            successors = [int(v) for v in np.where(graph[node] > 0)[0] if not req.scheduled[int(v)]]
            memo[node] = max((min_service(v) + path(v) for v in successors), default=0.0)
            return memo[node]

        return float(path(task_id))


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        mask = np.ones_like(mask, dtype=bool)
    safe_logits = np.where(mask, logits, -1e9)
    safe_logits = safe_logits - float(np.max(safe_logits))
    exp = np.exp(safe_logits) * mask
    total = float(np.sum(exp))
    if total <= 0:
        probs = mask.astype(np.float64) / max(1, int(np.sum(mask)))
    else:
        probs = (exp / total).astype(np.float64)
    probs = np.maximum(probs, 0.0)
    return probs / float(np.sum(probs))
