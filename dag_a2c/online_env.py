"""Online multi-request dual-DAG serving environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dag_a2c.structures import DualDAGInstance, llm_transition_components, model_state_reuse_vector
from dag_a2c.trace_generator import TEMPLATES, generate_trace_instance


@dataclass
class ActiveRequest:
    request_id: int
    arrival_time: float
    instance: DualDAGInstance
    completed: np.ndarray
    scheduled: np.ndarray
    task_model: np.ndarray
    task_server: np.ndarray
    finish_times: np.ndarray
    template_name: str = ""


class OnlineDualDAGServingEnv:
    """Simulate online arrivals of heterogeneous request DAGs.

    Action is ``(request_slot, task_id, model_id, server_id)``. Requests have
    already passed an exogenous front-door capacity check and use posted service
    tariffs. This environment studies the provider's runtime orchestration, not
    auction or personalized pricing.
    """

    def __init__(
        self,
        horizon: float = 80.0,
        arrival_rate: float = 0.32,
        max_active_requests: int = 6,
        max_task_slots: int = 6,
        max_steps: int = 260,
        seed: int | None = None,
        strict_model_dag: bool = False,
        deadline_tightness: float = 1.0,
        budget_scale: float = 1.0,
    ) -> None:
        self.horizon = float(horizon)
        self.arrival_rate = float(arrival_rate)
        self.max_active_requests = int(max_active_requests)
        self.max_task_slots = int(max_task_slots)
        self.max_steps = int(max_steps)
        self.strict_model_dag = strict_model_dag
        self.deadline_tightness = float(deadline_tightness)
        self.budget_scale = float(budget_scale)
        self.rng = np.random.default_rng(seed)
        probe = generate_trace_instance("longbench_rag_qa", seed=seed)
        self.models = probe.models
        self.servers = probe.servers
        self.model_graph = probe.model_graph
        self.model_reuse = probe.model_reuse
        self.transition_allowed = probe.transition_allowed
        self.num_models = len(self.models)
        self.num_servers = len(self.servers)
        self.num_tasks = self.max_task_slots
        self.reset(seed)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pending = self._generate_arrivals(seed)
        self.all_requests = list(self.pending)
        self.active: list[ActiveRequest] = []
        self.done_requests: list[ActiveRequest] = []
        self.running: list[dict[str, float | int]] = []
        self.current_time = 0.0
        self.step_count = 0
        self.server_time = np.zeros(self.num_servers, dtype=np.float32)
        self.server_memory = np.zeros(self.num_servers, dtype=np.float32)
        self.resident = np.zeros((self.num_servers, self.num_models), dtype=bool)
        self.cumulative_reward = 0.0
        self.cumulative_revenue = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_switch = 0.0
        self.cumulative_switch_components = {name: 0.0 for name in ("load", "tokenizer", "kv", "adapter", "context", "migration")}
        self.cumulative_violation = 0.0
        self.cumulative_sla_penalty = 0.0
        self.invalid = 0
        self.history: list[dict[str, float | int | str | bool]] = []
        self._release_arrivals()
        self._complete_due_tasks()
        return self.observe()

    @property
    def done(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if not self.pending and not self.active and not self.running:
            return True
        if self.current_time >= self.horizon and not self.pending and not self.running and not self.feasible_actions():
            return True
        return False

    def step(self, action: np.ndarray | tuple[int, int, int, int]) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        self.step_count += 1
        if not self.feasible_actions():
            self._advance_to_next_event()
            reward = 0.0
            self.cumulative_reward += reward
            return self.observe(), reward, self.done, {"event": "idle"}

        req_slot, task_id, model_id, server_id = np.asarray(action, dtype=np.int64).reshape(-1)[:4]
        if not self.is_action_valid(int(req_slot), int(task_id), int(model_id), int(server_id)):
            self.invalid += 1
            reward = -3.0
            self.cumulative_reward += reward
            self.history.append({"step": self.step_count, "event": "invalid", "reward": reward})
            return self.observe(), reward, self.done, {"event": "invalid"}

        req = self.active[int(req_slot)]
        instance = req.instance
        task = instance.request.tasks[int(task_id)]
        model = self.models[int(model_id)]
        duration = self.execution_duration(req, int(task_id), int(model_id), int(server_id))
        switch_components = self.transition_components(req, int(task_id), int(model_id), int(server_id))
        switch = switch_components["total"]
        start = max(float(self.server_time[int(server_id)]), self.current_time, req.arrival_time)
        finish = start + duration + switch

        execution_cost = task.compute_demand * model.unit_cost + duration * self.servers[int(server_id)].operating_cost_per_time
        transition_cost = switch * self.servers[int(server_id)].operating_cost_per_time
        resource_cost = execution_cost + transition_cost
        revenue = 0.0
        violation = 0.0
        sla_penalty = 0.0
        response_latency = finish - req.arrival_time

        req.scheduled[int(task_id)] = True
        req.task_model[int(task_id)] = int(model_id)
        req.task_server[int(task_id)] = int(server_id)
        req.finish_times[int(task_id)] = finish
        if bool(np.all(req.scheduled)):
            workflow_finish = float(np.nanmax(req.finish_times))
            absolute_deadline = req.arrival_time + instance.request.deadline
            violation = max(0.0, workflow_finish - absolute_deadline)
            if violation <= 1e-6:
                revenue = float(instance.request.service_fee)
            else:
                sla_penalty = float(instance.request.sla_penalty)
            self.cumulative_revenue += revenue
            self.cumulative_sla_penalty += sla_penalty
            self.cumulative_violation += violation
        reward = revenue - resource_cost - sla_penalty
        self.server_time[int(server_id)] = finish
        self.current_time = max(self.current_time, float(start))
        self.resident[int(server_id), int(model_id)] = True
        self.server_memory[int(server_id)] = float(sum(model.memory for model in self.models if self.resident[int(server_id), model.model_id]))
        self.cumulative_cost += resource_cost
        self.cumulative_switch += switch
        for name in self.cumulative_switch_components:
            self.cumulative_switch_components[name] += switch_components[name]
        self.cumulative_reward += reward
        self.history.append(
            {
                "step": self.step_count,
                "event": "schedule",
                "request": req.request_id,
                "template": req.template_name,
                "task": int(task_id),
                "model": int(model_id),
                "server": int(server_id),
                "service_fee": float(revenue),
                "start": float(start),
                "finish": float(finish),
                "latency": float(response_latency),
                "switch": float(switch),
                "load_overhead": float(switch_components["load"]),
                "tokenizer_overhead": float(switch_components["tokenizer"]),
                "kv_overhead": float(switch_components["kv"]),
                "adapter_overhead": float(switch_components["adapter"]),
                "context_overhead": float(switch_components["context"]),
                "migration_overhead": float(switch_components["migration"]),
                "violation": float(violation),
                "operational_cost": float(resource_cost),
                "sla_penalty": float(sla_penalty),
                "reward": float(reward),
            }
        )

        self.running.append(
            {
                "finish": float(finish),
                "request_id": req.request_id,
                "task": int(task_id),
            }
        )
        self._release_arrivals()
        return self.observe(), float(reward), self.done, {"event": "schedule"}

    def feasible_actions(self) -> list[tuple[int, int, int, int]]:
        actions: list[tuple[int, int, int, int]] = []
        for req_slot, req in enumerate(self.active[: self.max_active_requests]):
            for task_id in self.ready_tasks(req):
                for model_id in np.where(req.instance.compatibility[task_id] > 0)[0]:
                    for server_id in range(self.num_servers):
                        if self.is_action_valid(req_slot, task_id, int(model_id), server_id):
                            actions.append((req_slot, task_id, int(model_id), server_id))
        return actions

    def feasible_action_masks(self) -> dict[str, np.ndarray]:
        actions = self.feasible_actions()
        masks = {
            "request": np.zeros(self.max_active_requests, dtype=bool),
            "task": np.zeros(self.max_task_slots, dtype=bool),
            "model": np.zeros(self.num_models, dtype=bool),
            "server": np.zeros(self.num_servers, dtype=bool),
        }
        for req_slot, task_id, model_id, server_id in actions:
            masks["request"][req_slot] = True
            masks["task"][task_id] = True
            masks["model"][model_id] = True
            masks["server"][server_id] = True
        return masks

    def is_action_valid(self, req_slot: int, task_id: int, model_id: int, server_id: int) -> bool:
        if not (0 <= req_slot < len(self.active) and 0 <= task_id < self.max_task_slots and 0 <= model_id < self.num_models and 0 <= server_id < self.num_servers):
            return False
        req = self.active[req_slot]
        if task_id >= len(req.instance.request.tasks):
            return False
        if task_id not in self.ready_tasks(req):
            return False
        if req.instance.compatibility[task_id, model_id] <= 0:
            return False
        if self.strict_model_dag and not self.model_transition_allowed(req, task_id, model_id):
            return False
        projected = self.server_memory[server_id] if self.resident[server_id, model_id] else self.server_memory[server_id] + self.models[model_id].memory
        return bool(projected <= self.servers[server_id].memory_capacity + 1e-6)

    def ready_tasks(self, req: ActiveRequest) -> list[int]:
        ready: list[int] = []
        graph = req.instance.request.task_graph
        for task_id in range(len(req.instance.request.tasks)):
            if req.completed[task_id]:
                continue
            if req.scheduled[task_id]:
                continue
            preds = np.where(graph[:, task_id] > 0)[0]
            if np.all(req.completed[preds]):
                ready.append(task_id)
        return ready

    def model_transition_allowed(self, req: ActiveRequest, task_id: int, model_id: int) -> bool:
        allowed = req.instance.transition_allowed
        if allowed is None:
            return True
        preds = np.where(req.instance.request.task_graph[:, task_id] > 0)[0]
        for pred in preds:
            pred_model = int(req.task_model[pred])
            if pred_model >= 0 and allowed[pred_model, model_id] <= 0:
                return False
        return True

    def execution_duration(self, req: ActiveRequest, task_id: int, model_id: int, server_id: int) -> float:
        task = req.instance.request.tasks[task_id]
        model = self.models[model_id]
        server = self.servers[server_id]
        io = 0.06 * (task.input_size + task.output_size)
        return float(model.base_latency * req.instance.task_model_complexity[task_id, model_id] * server.latency_factor + task.compute_demand / server.compute_capacity + io)

    def transition_overhead(self, req: ActiveRequest, task_id: int, model_id: int, server_id: int) -> float:
        return float(self.transition_components(req, task_id, model_id, server_id)["total"])

    def transition_components(self, req: ActiveRequest, task_id: int, model_id: int, server_id: int) -> dict[str, float]:
        model = self.models[model_id]
        server = self.servers[server_id]
        task = req.instance.request.tasks[task_id]
        preds = np.where(req.instance.request.task_graph[:, task_id] > 0)[0]
        if len(preds) == 0:
            comps = llm_transition_components(None, model, task, server, resident=bool(self.resident[server_id, model_id]), has_predecessor_state=False)
            comps["migration"] = 0.0
            return comps
        total = {name: 0.0 for name in ("load", "tokenizer", "kv", "adapter", "context", "migration", "total")}
        first = True
        for pred in preds:
            pred_model = int(req.task_model[pred])
            if pred_model >= 0:
                pred_server = int(req.task_server[pred])
                comps = llm_transition_components(
                    self.models[pred_model],
                    model,
                    task,
                    server,
                    resident=bool(self.resident[server_id, model_id]) or not first,
                    has_predecessor_state=True,
                )
                migration = 0.0
                if pred_server >= 0 and pred_server != server_id:
                    _, rho_kv, _, rho_ctx = model_state_reuse_vector(self.models[pred_model], model, task)
                    transferable_state = rho_kv * task.kv_cache_size + rho_ctx * task.context_state_size
                    path_bandwidth = min(self.servers[pred_server].bandwidth, server.bandwidth)
                    migration = transferable_state / max(path_bandwidth, 1e-6)
                comps["migration"] = float(migration)
                comps["total"] += float(migration)
                for name, value in comps.items():
                    total[name] += value
                first = False
        return {name: float(value) for name, value in total.items()}

    def observe(self) -> dict[str, np.ndarray]:
        request_features = np.zeros((self.max_active_requests, 8), dtype=np.float32)
        task_features = np.zeros((self.max_active_requests, self.max_task_slots, 10), dtype=np.float32)
        max_fee = max([req.instance.request.service_fee for req in self.active] + [1.0])
        for slot, req in enumerate(self.active[: self.max_active_requests]):
            ready = self.ready_tasks(req)
            completed_ratio = float(np.mean(req.completed))
            age = max(0.0, self.current_time - req.arrival_time)
            slack = req.arrival_time + req.instance.request.deadline - self.current_time
            request_features[slot] = np.array(
                [
                    1.0,
                    completed_ratio,
                    len(ready) / max(1, len(req.instance.request.tasks)),
                    np.clip(age / max(req.instance.request.deadline, 1e-6), 0.0, 2.0) / 2.0,
                    np.clip(slack / max(req.instance.request.deadline, 1e-6), 0.0, 1.0),
                    req.instance.request.service_fee / max_fee,
                    float(np.mean(req.scheduled)),
                    len(req.instance.request.tasks) / self.max_task_slots,
                ],
                dtype=np.float32,
            )
            max_compute = max(task.compute_demand for task in req.instance.request.tasks)
            max_io = max(task.input_size + task.output_size for task in req.instance.request.tasks)
            depths = self._remaining_depths(req)
            for task in req.instance.request.tasks:
                tid = task.task_id
                preds = np.where(req.instance.request.task_graph[:, tid] > 0)[0]
                succs = np.where(req.instance.request.task_graph[tid] > 0)[0]
                task_features[slot, tid] = np.array(
                    [
                        1.0,
                        float(req.completed[tid]),
                        float(tid in ready),
                        task.compute_demand / max_compute,
                        (task.input_size + task.output_size) / max_io,
                        task.min_quality,
                        len(preds) / max(1, self.max_task_slots - 1),
                        len(succs) / max(1, self.max_task_slots - 1),
                        depths[tid] / max(1.0, float(np.max(depths))),
                        np.sum(req.instance.compatibility[tid]) / self.num_models,
                    ],
                    dtype=np.float32,
                )

        model_features = np.zeros((self.num_models, 9), dtype=np.float32)
        max_cost = max(model.unit_cost for model in self.models)
        max_latency = max(model.base_latency for model in self.models)
        max_mem = max(model.memory for model in self.models)
        max_weight = max(model.weight_size for model in self.models)
        for model in self.models:
            mid = model.model_id
            model_features[mid] = np.array(
                [
                    model.quality,
                    model.unit_cost / max_cost,
                    model.base_latency / max_latency,
                    model.memory / max_mem,
                    model.weight_size / max_weight,
                    float(np.any(self.resident[:, mid])),
                    np.sum(self.model_graph[:, mid]) / max(1, self.num_models - 1),
                    np.sum(self.model_graph[mid]) / max(1, self.num_models - 1),
                    float(np.mean([req.instance.compatibility[:, mid].mean() for req in self.active])) if self.active else 0.0,
                ],
                dtype=np.float32,
            )

        server_features = np.zeros((self.num_servers, 5), dtype=np.float32)
        max_compute = max(server.compute_capacity for server in self.servers)
        max_server_mem = max(server.memory_capacity for server in self.servers)
        max_bandwidth = max(server.bandwidth for server in self.servers)
        for server in self.servers:
            sid = server.server_id
            server_features[sid] = np.array(
                [
                    server.compute_capacity / max_compute,
                    server.memory_capacity / max_server_mem,
                    server.bandwidth / max_bandwidth,
                    np.clip(self.server_memory[sid] / server.memory_capacity, 0.0, 1.0),
                    np.clip(max(0.0, self.server_time[sid] - self.current_time) / max(self.horizon, 1e-6), 0.0, 1.0),
                ],
                dtype=np.float32,
            )

        global_features = np.array(
            [
                np.clip(self.current_time / self.horizon, 0.0, 1.0),
                len(self.active) / self.max_active_requests,
                len(self.pending) / max(1, len(self.pending) + len(self.done_requests) + len(self.active)),
                len(self.done_requests) / max(1, len(self.all_requests)),
                np.clip(self.cumulative_revenue / 500.0, 0.0, 1.0),
                np.clip(self.cumulative_cost / 200.0, 0.0, 1.0),
                np.clip(self.cumulative_switch / 80.0, 0.0, 1.0),
                np.clip(self.cumulative_violation / 80.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return {
            "request_features": request_features,
            "task_features": task_features,
            "model_features": model_features,
            "server_features": server_features,
            "global_features": global_features,
            "model_graph": self.model_graph.astype(np.float32),
            "model_reuse": self.model_reuse.astype(np.float32) if self.model_reuse is not None else np.zeros((self.num_models, self.num_models, 4), dtype=np.float32),
        }

    def state_vector(self) -> np.ndarray:
        obs = self.observe()
        return np.concatenate([value.reshape(-1) for value in obs.values()]).astype(np.float32)

    def final_metrics(self) -> dict[str, float]:
        completed = len(self.done_requests)
        total_arrived = len(getattr(self, "all_requests", []))
        latencies = []
        for req in self.done_requests:
            latencies.append(float(np.nanmax(req.finish_times) - req.arrival_time))
        metrics = {
            "episode_return": float(self.cumulative_reward),
            "requests_arrived": float(total_arrived),
            "requests_completed": float(completed),
            "request_completion_rate": completed / max(1, total_arrived),
            "admission_rate": 1.0,
            "avg_request_latency": float(np.mean(latencies)) if latencies else 0.0,
            "p95_request_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "revenue": float(self.cumulative_revenue),
            "cost": float(self.cumulative_cost),
            "profit": float(self.cumulative_revenue - self.cumulative_cost - self.cumulative_sla_penalty),
            "sla_penalty": float(self.cumulative_sla_penalty),
            "switching_overhead": float(self.cumulative_switch),
            "load_overhead": float(self.cumulative_switch_components["load"]),
            "tokenizer_overhead": float(self.cumulative_switch_components["tokenizer"]),
            "kv_overhead": float(self.cumulative_switch_components["kv"]),
            "adapter_overhead": float(self.cumulative_switch_components["adapter"]),
            "context_overhead": float(self.cumulative_switch_components["context"]),
            "migration_overhead": float(self.cumulative_switch_components["migration"]),
            "deadline_violation": float(self.cumulative_violation),
            "invalid_actions": float(self.invalid),
            "rejected_requests": 0.0,
            "steps": float(self.step_count),
        }
        for template in TEMPLATES:
            arrived = sum(1 for req in getattr(self, "all_requests", []) if req.template_name == template)
            completed_template = sum(1 for req in self.done_requests if req.template_name == template)
            metrics[f"arrived_{template}"] = float(arrived)
            metrics[f"completed_{template}"] = float(completed_template)
            metrics[f"completion_rate_{template}"] = completed_template / max(1, arrived)
        return metrics

    def _generate_arrivals(self, seed: int | None) -> list[ActiveRequest]:
        rng = np.random.default_rng(seed)
        arrivals: list[ActiveRequest] = []
        time = 0.0
        request_id = 0
        template_names = tuple(TEMPLATES.keys())
        while time < self.horizon:
            time += float(rng.exponential(1.0 / max(self.arrival_rate, 1e-6)))
            if time >= self.horizon:
                break
            template = str(rng.choice(template_names, p=np.array([0.22, 0.34, 0.24, 0.20])))
            instance = generate_trace_instance(template, seed=int(rng.integers(1, 1_000_000)), deadline_tightness=self.deadline_tightness, budget_scale=self.budget_scale)
            arrivals.append(
                ActiveRequest(
                    request_id=request_id,
                    arrival_time=time,
                    instance=instance,
                    completed=np.zeros(len(instance.request.tasks), dtype=bool),
                    scheduled=np.zeros(len(instance.request.tasks), dtype=bool),
                    task_model=np.full(len(instance.request.tasks), -1, dtype=np.int64),
                    task_server=np.full(len(instance.request.tasks), -1, dtype=np.int64),
                    finish_times=np.full(len(instance.request.tasks), np.nan, dtype=np.float32),
                    template_name=template,
                )
            )
            request_id += 1
        return arrivals

    def _release_arrivals(self) -> None:
        while self.pending and self.pending[0].arrival_time <= self.current_time and len(self.active) < self.max_active_requests:
            self.active.append(self.pending.pop(0))

    def _advance_to_next_event(self) -> None:
        next_times = []
        if self.pending and len(self.active) < self.max_active_requests:
            next_times.append(self.pending[0].arrival_time)
        busy = self.server_time[self.server_time > self.current_time + 1e-6]
        if len(busy):
            next_times.append(float(np.min(busy)))
        if self.running:
            next_times.append(float(min(item["finish"] for item in self.running)))
        if next_times:
            self.current_time = min(next_times)
            self._complete_due_tasks()
            self._release_arrivals()
        else:
            self.current_time = self.horizon

    def _complete_due_tasks(self) -> None:
        if not self.running:
            return
        due = [item for item in self.running if float(item["finish"]) <= self.current_time + 1e-6]
        self.running = [item for item in self.running if float(item["finish"]) > self.current_time + 1e-6]
        if not due:
            return
        completed_ids: set[int] = set()
        for item in due:
            request_id = int(item["request_id"])
            task_id = int(item["task"])
            for req in self.active:
                if req.request_id == request_id:
                    req.completed[task_id] = True
                    if bool(np.all(req.completed)):
                        completed_ids.add(req.request_id)
                    break
        if completed_ids:
            remaining: list[ActiveRequest] = []
            for req in self.active:
                if req.request_id in completed_ids:
                    self.done_requests.append(req)
                else:
                    remaining.append(req)
            self.active = remaining

    def _remaining_depths(self, req: ActiveRequest) -> np.ndarray:
        graph = req.instance.request.task_graph
        depths = np.zeros(len(req.instance.request.tasks), dtype=np.float32)
        for node in range(len(depths) - 1, -1, -1):
            succs = np.where(graph[node] > 0)[0]
            if len(succs):
                depths[node] = 1.0 + float(np.max(depths[succs]))
        return depths
