"""Pure NumPy environment for dual-DAG scheduling, model selection, and pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dag_a2c.generator import generate_dual_dag_instance
from dag_a2c.structures import DualDAGInstance, llm_transition_components


@dataclass(frozen=True)
class RewardWeights:
    cost: float = 1.0
    latency: float = 0.14
    violation: float = 1.25
    switching: float = 0.9
    quality_bonus: float = 0.25


class DualDAGServingEnv:
    """One-request dual-DAG serving simulator.

    Action is a 4-tuple: ``(ready_task_id, model_id, server_id, price_bin_id)``.
    The simulator enforces task-DAG readiness and task-model compatibility,
    then charges transition overhead according to the selected models on each
    task-DAG edge and the provider-side model DAG.
    """

    def __init__(
        self,
        instance: DualDAGInstance | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
        strict_model_dag: bool = False,
        invalid_action_penalty: float = 3.0,
        weights: RewardWeights | None = None,
    ) -> None:
        self.instance = instance or generate_dual_dag_instance(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.strict_model_dag = strict_model_dag
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.weights = weights or RewardWeights()
        self.max_steps = max_steps or self.instance.num_tasks * 6
        self.num_tasks = self.instance.num_tasks
        self.num_models = self.instance.num_models
        self.num_servers = self.instance.num_servers
        self.num_prices = len(self.instance.price_bins)

        self._task_depth = self._remaining_depths()
        self._max_compute = max(task.compute_demand for task in self.instance.request.tasks)
        self._max_io = max(task.input_size + task.output_size for task in self.instance.request.tasks)
        self._max_mem = max(model.memory for model in self.instance.models)
        self._max_price = float(np.max(self.instance.price_bins))
        self._max_time = max(self.instance.request.deadline * 1.5, 1.0)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.completed = np.zeros(self.num_tasks, dtype=bool)
        self.task_model = np.full(self.num_tasks, -1, dtype=np.int64)
        self.task_server = np.full(self.num_tasks, -1, dtype=np.int64)
        self.finish_times = np.full(self.num_tasks, np.nan, dtype=np.float32)
        self.server_time = np.zeros(self.num_servers, dtype=np.float32)
        self.server_memory = np.zeros(self.num_servers, dtype=np.float32)
        self.resident = np.zeros((self.num_servers, self.num_models), dtype=bool)
        self.current_time = 0.0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.cumulative_revenue = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_switch = 0.0
        self.cumulative_switch_components = {name: 0.0 for name in ("load", "tokenizer", "kv", "adapter", "context")}
        self.cumulative_violation = 0.0
        self.accepted = 0
        self.rejected = 0
        self.invalid = 0
        self.history: list[dict[str, float | int | str | bool]] = []
        return self.observe()

    def step(self, action: np.ndarray | tuple[int, int, int, int] | list[int]) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        task_id, model_id, server_id, price_id = np.asarray(action, dtype=np.int64).reshape(-1)[:4]
        self.step_count += 1
        info: dict[str, Any] = {}

        if not self.is_action_valid(int(task_id), int(model_id), int(server_id), int(price_id)):
            self.invalid += 1
            self.current_time += 0.20
            reward = -self.invalid_action_penalty
            self.cumulative_reward += reward
            self.history.append(
                {
                    "step": self.step_count,
                    "event": "invalid",
                    "task": int(task_id),
                    "model": int(model_id),
                    "server": int(server_id),
                    "price_id": int(price_id),
                    "reward": float(reward),
                }
            )
            done = self.done
            info.update(self.step_info(False, "invalid"))
            return self.observe(), reward, done, info

        task = self.instance.request.tasks[int(task_id)]
        model = self.instance.models[int(model_id)]
        server = self.instance.servers[int(server_id)]
        price = float(self.instance.price_bins[int(price_id)])

        duration = self.execution_duration(int(task_id), int(model_id), int(server_id))
        switch_components = self.transition_components(int(task_id), int(model_id), int(server_id))
        switch_overhead = switch_components["total"]
        expected_latency = max(float(self.server_time[int(server_id)]), self.current_time) + duration + switch_overhead
        expected_quality = float(model.quality)
        accepted = self.user_accepts(price, expected_latency, expected_quality)
        if not accepted:
            self.rejected += 1
            self.current_time += 0.25
            reward = -1.25
            self.cumulative_reward += reward
            self.history.append(
                {
                    "step": self.step_count,
                    "event": "rejected",
                    "task": int(task_id),
                    "model": int(model_id),
                    "server": int(server_id),
                    "price_id": int(price_id),
                    "price": price,
                    "expected_latency": float(expected_latency),
                    "quality": expected_quality,
                    "reward": float(reward),
                }
            )
            done = self.done
            info.update(self.step_info(False, "rejected", price=price))
            return self.observe(), reward, done, info

        start = max(float(self.server_time[int(server_id)]), self.current_time)
        finish = start + duration + switch_overhead
        resource_cost = task.compute_demand * model.unit_cost
        violation = max(0.0, finish - self.instance.request.deadline)
        latency = finish
        reward = (
            price
            - self.weights.cost * resource_cost
            - self.weights.latency * latency
            - self.weights.violation * violation
            - self.weights.switching * switch_overhead
            + self.weights.quality_bonus * task.compute_demand * model.quality
        )

        self.completed[int(task_id)] = True
        self.task_model[int(task_id)] = int(model_id)
        self.task_server[int(task_id)] = int(server_id)
        self.finish_times[int(task_id)] = finish
        self.server_time[int(server_id)] = finish
        self.current_time = float(np.min(self.server_time))
        self.resident[int(server_id), int(model_id)] = True
        self.server_memory[int(server_id)] = float(np.sum([m.memory for m in self.instance.models if self.resident[int(server_id), m.model_id]]))
        self.accepted += 1
        self.cumulative_revenue += price
        self.cumulative_cost += resource_cost
        self.cumulative_switch += switch_overhead
        for name in self.cumulative_switch_components:
            self.cumulative_switch_components[name] += switch_components[name]
        self.cumulative_violation += violation
        self.cumulative_reward += reward
        self.history.append(
            {
                "step": self.step_count,
                "event": "accepted",
                "task": int(task_id),
                "model": int(model_id),
                "server": int(server_id),
                "price_id": int(price_id),
                "price": price,
                "start": float(start),
                "finish": float(finish),
                "duration": float(duration),
                "switch_overhead": float(switch_overhead),
                "load_overhead": float(switch_components["load"]),
                "tokenizer_overhead": float(switch_components["tokenizer"]),
                "kv_overhead": float(switch_components["kv"]),
                "adapter_overhead": float(switch_components["adapter"]),
                "context_overhead": float(switch_components["context"]),
                "resource_cost": float(resource_cost),
                "violation": float(violation),
                "reward": float(reward),
            }
        )

        done = self.done
        info.update(
            self.step_info(
                True,
                "accepted",
                price=price,
                duration=duration,
                switch_overhead=switch_overhead,
                transition_components=switch_components,
                resource_cost=resource_cost,
                violation=violation,
            )
        )
        return self.observe(), float(reward), done, info

    @property
    def done(self) -> bool:
        if bool(np.all(self.completed)):
            return True
        if self.step_count >= self.max_steps:
            return True
        return not self.feasible_actions()

    def feasible_actions(self) -> list[tuple[int, int, int, int]]:
        actions: list[tuple[int, int, int, int]] = []
        for task_id in self.ready_tasks():
            for model_id in np.where(self.instance.compatibility[task_id] > 0)[0]:
                for server_id in range(self.num_servers):
                    for price_id in range(self.num_prices):
                        if self.is_action_valid(task_id, int(model_id), server_id, price_id):
                            actions.append((task_id, int(model_id), server_id, price_id))
        return actions

    def ready_tasks(self) -> list[int]:
        ready: list[int] = []
        for task_id in range(self.num_tasks):
            if self.completed[task_id]:
                continue
            preds = np.where(self.instance.request.task_graph[:, task_id] > 0)[0]
            if np.all(self.completed[preds]):
                ready.append(task_id)
        return ready

    def feasible_action_masks(self) -> dict[str, np.ndarray]:
        task_mask = np.zeros(self.num_tasks, dtype=bool)
        task_mask[self.ready_tasks()] = True
        model_mask = np.any(self.instance.compatibility[task_mask], axis=0) if np.any(task_mask) else np.zeros(self.num_models, dtype=bool)
        server_mask = np.array([server.memory_capacity > 0 for server in self.instance.servers], dtype=bool)
        price_mask = np.ones(self.num_prices, dtype=bool)
        return {"task": task_mask, "model": model_mask, "server": server_mask, "price": price_mask}

    def is_action_valid(self, task_id: int, model_id: int, server_id: int, price_id: int) -> bool:
        if not (0 <= task_id < self.num_tasks and 0 <= model_id < self.num_models and 0 <= server_id < self.num_servers and 0 <= price_id < self.num_prices):
            return False
        if task_id not in self.ready_tasks():
            return False
        if self.instance.compatibility[task_id, model_id] <= 0:
            return False
        if self.strict_model_dag and not self.model_transition_allowed(task_id, model_id):
            return False
        model = self.instance.models[model_id]
        server = self.instance.servers[server_id]
        projected = self.server_memory[server_id] if self.resident[server_id, model_id] else self.server_memory[server_id] + model.memory
        return bool(projected <= server.memory_capacity + 1e-6)

    def model_transition_allowed(self, task_id: int, model_id: int) -> bool:
        allowed = self.instance.transition_allowed
        if allowed is None:
            return True
        preds = np.where(self.instance.request.task_graph[:, task_id] > 0)[0]
        for pred in preds:
            pred_model = int(self.task_model[pred])
            if pred_model >= 0 and allowed[pred_model, model_id] <= 0:
                return False
        return True

    def execution_duration(self, task_id: int, model_id: int, server_id: int) -> float:
        task = self.instance.request.tasks[task_id]
        model = self.instance.models[model_id]
        server = self.instance.servers[server_id]
        io = 0.06 * (task.input_size + task.output_size)
        return float(model.base_latency * self.instance.task_model_complexity[task_id, model_id] * server.latency_factor + task.compute_demand / server.compute_capacity + io)

    def transition_overhead(self, task_id: int, model_id: int, server_id: int) -> float:
        return float(self.transition_components(task_id, model_id, server_id)["total"])

    def transition_components(self, task_id: int, model_id: int, server_id: int) -> dict[str, float]:
        model = self.instance.models[model_id]
        server = self.instance.servers[server_id]
        task = self.instance.request.tasks[task_id]
        preds = np.where(self.instance.request.task_graph[:, task_id] > 0)[0]
        if len(preds) == 0:
            return llm_transition_components(None, model, task, server, resident=bool(self.resident[server_id, model_id]), has_predecessor_state=False)
        total = {name: 0.0 for name in ("load", "tokenizer", "kv", "adapter", "context", "total")}
        first = True
        for pred in preds:
            pred_model = int(self.task_model[pred])
            if pred_model >= 0:
                comps = llm_transition_components(
                    self.instance.models[pred_model],
                    model,
                    task,
                    server,
                    resident=bool(self.resident[server_id, model_id]) or not first,
                    has_predecessor_state=True,
                )
                for name, value in comps.items():
                    total[name] += value
                first = False
        return {name: float(value) for name, value in total.items()}

    def user_accepts(self, price: float, latency: float, quality: float) -> bool:
        probability = self.acceptance_probability(price, latency, quality)
        return bool(self.rng.random() < probability)

    def acceptance_probability(self, price: float, latency: float, quality: float) -> float:
        req = self.instance.request
        utility = req.valuation + req.alpha_quality * quality - req.beta_latency * latency - price
        if price > req.budget:
            utility -= 1.8 * (price - req.budget)
        probability = 1.0 / (1.0 + np.exp(-utility / max(0.25 * req.budget, 1e-6)))
        return float(np.clip(probability, 0.02, 0.98))

    def observe(self) -> dict[str, np.ndarray]:
        task_features = np.zeros((self.num_tasks, 10), dtype=np.float32)
        ready = set(self.ready_tasks())
        for task in self.instance.request.tasks:
            tid = task.task_id
            preds = np.where(self.instance.request.task_graph[:, tid] > 0)[0]
            succs = np.where(self.instance.request.task_graph[tid] > 0)[0]
            slack = max(0.0, self.instance.request.deadline - self.current_time)
            feasible_models = float(np.sum(self.instance.compatibility[tid]))
            task_features[tid] = np.array(
                [
                    float(self.completed[tid]),
                    float(tid in ready),
                    task.compute_demand / self._max_compute,
                    (task.input_size + task.output_size) / self._max_io,
                    task.min_quality,
                    len(preds) / max(1, self.num_tasks - 1),
                    len(succs) / max(1, self.num_tasks - 1),
                    self._task_depth[tid] / max(1.0, float(np.max(self._task_depth))),
                    feasible_models / max(1, self.num_models),
                    np.clip(slack / self.instance.request.deadline, 0.0, 1.0),
                ],
                dtype=np.float32,
            )

        model_features = np.zeros((self.num_models, 9), dtype=np.float32)
        max_cost = max(model.unit_cost for model in self.instance.models)
        max_latency = max(model.base_latency for model in self.instance.models)
        max_weight = max(model.weight_size for model in self.instance.models)
        for model in self.instance.models:
            mid = model.model_id
            model_features[mid] = np.array(
                [
                    model.quality,
                    model.unit_cost / max_cost,
                    model.base_latency / max_latency,
                    model.memory / self._max_mem,
                    model.weight_size / max_weight,
                    float(np.any(self.resident[:, mid])),
                    np.sum(self.instance.model_graph[:, mid]) / max(1, self.num_models - 1),
                    np.sum(self.instance.model_graph[mid]) / max(1, self.num_models - 1),
                    np.sum(self.instance.compatibility[:, mid]) / max(1, self.num_tasks),
                ],
                dtype=np.float32,
            )

        server_features = np.zeros((self.num_servers, 5), dtype=np.float32)
        max_compute = max(server.compute_capacity for server in self.instance.servers)
        max_server_mem = max(server.memory_capacity for server in self.instance.servers)
        max_bandwidth = max(server.bandwidth for server in self.instance.servers)
        for server in self.instance.servers:
            sid = server.server_id
            server_features[sid] = np.array(
                [
                    server.compute_capacity / max_compute,
                    server.memory_capacity / max_server_mem,
                    server.bandwidth / max_bandwidth,
                    np.clip(self.server_memory[sid] / server.memory_capacity, 0.0, 1.0),
                    np.clip(self.server_time[sid] / self._max_time, 0.0, 1.0),
                ],
                dtype=np.float32,
            )

        global_features = np.array(
            [
                np.mean(self.completed),
                len(ready) / max(1, self.num_tasks),
                np.clip(self.current_time / self._max_time, 0.0, 1.0),
                np.clip(self.instance.request.budget / self._max_price, 0.0, 1.0),
                np.clip(self.cumulative_revenue / max(self.instance.request.budget, 1e-6), 0.0, 2.0) / 2.0,
                np.clip(self.cumulative_cost / max(self.instance.request.budget, 1e-6), 0.0, 2.0) / 2.0,
                np.clip(self.cumulative_switch / self._max_time, 0.0, 1.0),
                np.clip(self.cumulative_violation / self._max_time, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return {
            "task_features": task_features,
            "model_features": model_features,
            "server_features": server_features,
            "global_features": global_features,
            "task_graph": self.instance.request.task_graph.astype(np.float32),
            "model_graph": self.instance.model_graph.astype(np.float32),
            "model_reuse": self.instance.model_reuse.astype(np.float32) if self.instance.model_reuse is not None else np.zeros((self.num_models, self.num_models, 4), dtype=np.float32),
        }

    def state_vector(self) -> np.ndarray:
        obs = self.observe()
        return np.concatenate(
            [
                obs["task_features"].reshape(-1),
                obs["model_features"].reshape(-1),
                obs["server_features"].reshape(-1),
                obs["global_features"].reshape(-1),
                obs["task_graph"].reshape(-1),
                obs["model_graph"].reshape(-1),
                obs["model_reuse"].reshape(-1),
            ]
        ).astype(np.float32)

    def final_metrics(self) -> dict[str, float]:
        completed = int(np.sum(self.completed))
        finished = self.finish_times[~np.isnan(self.finish_times)]
        makespan = float(np.max(finished)) if len(finished) else 0.0
        return {
            "episode_return": float(self.cumulative_reward),
            "completed_tasks": float(completed),
            "completion_rate": completed / max(1, self.num_tasks),
            "acceptance_rate": self.accepted / max(1, self.accepted + self.rejected),
            "revenue": float(self.cumulative_revenue),
            "cost": float(self.cumulative_cost),
            "profit": float(self.cumulative_revenue - self.cumulative_cost),
            "switching_overhead": float(self.cumulative_switch),
            "load_overhead": float(self.cumulative_switch_components["load"]),
            "tokenizer_overhead": float(self.cumulative_switch_components["tokenizer"]),
            "kv_overhead": float(self.cumulative_switch_components["kv"]),
            "adapter_overhead": float(self.cumulative_switch_components["adapter"]),
            "context_overhead": float(self.cumulative_switch_components["context"]),
            "deadline_violation": float(self.cumulative_violation),
            "makespan": makespan,
            "invalid_actions": float(self.invalid),
            "rejected_actions": float(self.rejected),
            "steps": float(self.step_count),
        }

    def step_info(self, accepted: bool, event: str, **extra: Any) -> dict[str, Any]:
        info = {
            "event": event,
            "accepted": accepted,
            "ready_tasks": len(self.ready_tasks()),
            **extra,
        }
        if self.done:
            info["final_metrics"] = self.final_metrics()
        return info

    def _remaining_depths(self) -> np.ndarray:
        graph = self.instance.request.task_graph
        depths = np.zeros(self.instance.num_tasks, dtype=np.float32)
        for node in range(self.instance.num_tasks - 1, -1, -1):
            succs = np.where(graph[node] > 0)[0]
            if len(succs):
                depths[node] = 1.0 + float(np.max(depths[succs]))
        return depths
