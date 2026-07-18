"""Baselines and evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dag_a2c.env import DualDAGServingEnv

PolicyFn = Callable[[DualDAGServingEnv], np.ndarray]


def random_feasible_policy(env: DualDAGServingEnv) -> np.ndarray:
    valid = env.feasible_actions()
    if not valid:
        return np.zeros(4, dtype=np.int64)
    return np.asarray(valid[int(env.rng.integers(0, len(valid)))], dtype=np.int64)


def greedy_dual_dag_policy(env: DualDAGServingEnv) -> np.ndarray:
    """Myopic utility heuristic that uses both task and model DAGs."""

    best_action: np.ndarray | None = None
    best_score = -np.inf
    for task_id in env.ready_tasks():
        task = env.instance.request.tasks[task_id]
        for model_id in np.where(env.instance.compatibility[task_id] > 0)[0]:
            model = env.instance.models[int(model_id)]
            for server_id in range(env.num_servers):
                affordable = np.where(env.instance.price_bins <= env.instance.request.budget)[0]
                price_candidates = affordable if len(affordable) else np.arange(env.num_prices)
                for price_id in price_candidates[-3:]:
                    if not env.is_action_valid(task_id, int(model_id), server_id, int(price_id)):
                        continue
                    duration = env.execution_duration(task_id, int(model_id), server_id)
                    switch = env.transition_overhead(task_id, int(model_id), server_id)
                    finish = max(float(env.server_time[server_id]), env.current_time) + duration + switch
                    price = float(env.instance.price_bins[int(price_id)])
                    cost = task.compute_demand * model.unit_cost
                    violation = max(0.0, finish - env.instance.request.deadline)
                    criticality = env.observe()["task_features"][task_id, 7]
                    score = price + task.compute_demand * model.quality - cost - 0.30 * finish - switch - 1.4 * violation + criticality
                    if score > best_score:
                        best_score = score
                        best_action = np.array([task_id, int(model_id), server_id, int(price_id)], dtype=np.int64)
    return best_action if best_action is not None else random_feasible_policy(env)


def evaluate_policy(
    env_factory: Callable[[int], DualDAGServingEnv],
    policy: PolicyFn,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    metrics: list[dict[str, float]] = []
    for ep in range(episodes):
        env = env_factory(seed + ep)
        env.reset(seed + ep)
        while not env.done:
            env.step(policy(env))
        metrics.append(env.final_metrics())
    return average_metrics(metrics)


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    return {key: float(np.mean([row[key] for row in metrics])) for key in sorted(metrics[0])}
