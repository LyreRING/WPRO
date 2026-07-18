"""Small masked A2C implementation with NumPy only.

This is intentionally lightweight so the Windows workspace can run the full
algorithm without PyTorch or Stable-Baselines3. The policy is a factored actor
with four categorical heads: task, model, server, and price. The critic is a
linear value function over the same dual-DAG state vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dag_a2c.env import DualDAGServingEnv


@dataclass
class A2CConfig:
    gamma: float = 0.96
    actor_lr: float = 0.018
    critic_lr: float = 0.045
    entropy_coef: float = 0.010
    action_prior_coef: float = 0.75
    seed: int = 42


class MaskedLinearA2C:
    def __init__(self, state_dim: int, action_dims: tuple[int, int, int, int], config: A2CConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.action_dims = action_dims
        self.actor_w = [self.rng.normal(0.0, 0.015, size=(state_dim, dim)).astype(np.float32) for dim in action_dims]
        self.actor_b = [np.zeros(dim, dtype=np.float32) for dim in action_dims]
        self.critic_w = np.zeros(state_dim, dtype=np.float32)
        self.critic_b = 0.0

    def act(self, state: np.ndarray, env: DualDAGServingEnv, deterministic: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
        masks = self._head_masks(env)
        self.last_masks = [mask.copy() for mask in masks]
        probs: list[np.ndarray] = []
        for head, mask in enumerate(masks):
            logits = state @ self.actor_w[head] + self.actor_b[head]
            prob = masked_softmax(logits, mask)
            probs.append(prob)
        action = list(self._select_joint_action(state, env, deterministic))
        return np.asarray(action, dtype=np.int64), probs

    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool, env: DualDAGServingEnv) -> float:
        value = self.value(state)
        target = reward if done else reward + self.config.gamma * self.value(next_state)
        advantage = float(target - value)

        self.critic_w += self.config.critic_lr * advantage * state
        self.critic_b += self.config.critic_lr * advantage

        masks = getattr(self, "last_masks", self._head_masks(env))
        for head, dim_action in enumerate(action):
            logits = state @ self.actor_w[head] + self.actor_b[head]
            probs = masked_softmax(logits, masks[head])
            grad_logits = -probs
            grad_logits[int(dim_action)] += 1.0
            entropy_grad = -probs * (np.log(np.maximum(probs, 1e-8)) + 1.0)
            grad_logits = advantage * grad_logits + self.config.entropy_coef * entropy_grad
            self.actor_w[head] += self.config.actor_lr * np.outer(state, grad_logits)
            self.actor_b[head] += self.config.actor_lr * grad_logits
        return advantage

    def value(self, state: np.ndarray) -> float:
        return float(state @ self.critic_w + self.critic_b)

    def _head_masks(self, env: DualDAGServingEnv) -> list[np.ndarray]:
        masks = env.feasible_action_masks()
        return [masks["task"], masks["model"], masks["server"], masks["price"]]

    def _select_joint_action(self, state: np.ndarray, env: DualDAGServingEnv, deterministic: bool) -> tuple[int, int, int, int]:
        logits = [state @ self.actor_w[idx] + self.actor_b[idx] for idx in range(4)]
        actions = env.feasible_actions()
        if not actions:
            return (0, 0, 0, 0)
        scores = []
        for task_id, model_id, server_id, price_id in env.feasible_actions():
            score = logits[0][task_id] + logits[1][model_id] + logits[2][server_id] + logits[3][price_id]
            score += self.config.action_prior_coef * self._action_prior(env, task_id, model_id, server_id, price_id)
            scores.append(float(score))
        scores_arr = np.asarray(scores, dtype=np.float64)
        if deterministic:
            return actions[int(np.argmax(scores_arr))]
        probs = masked_softmax(scores_arr, np.ones(len(scores_arr), dtype=bool))
        return actions[int(self.rng.choice(len(actions), p=probs))]

    def _action_prior(self, env: DualDAGServingEnv, task_id: int, model_id: int, server_id: int, price_id: int) -> float:
        task = env.instance.request.tasks[task_id]
        model = env.instance.models[model_id]
        price = float(env.instance.price_bins[price_id])
        duration = env.execution_duration(task_id, model_id, server_id)
        switch = env.transition_overhead(task_id, model_id, server_id)
        finish = max(float(env.server_time[server_id]), env.current_time) + duration + switch
        cost = task.compute_demand * model.unit_cost
        violation = max(0.0, finish - env.instance.request.deadline)
        quality = float(model.quality)
        accept_prob = env.acceptance_probability(price, finish, quality)
        expected_price = accept_prob * price - (1.0 - accept_prob) * 2.0
        budget_penalty = max(0.0, price - env.instance.request.budget) * 1.4
        return float(
            expected_price
            - cost
            - 0.18 * finish
            - 1.20 * switch
            - 2.20 * violation
            - budget_penalty
            + 0.35 * task.compute_demand * quality
        )


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
    probs = probs / float(np.sum(probs))
    return probs
