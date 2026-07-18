"""Synthetic dual-DAG instance generation.

The generator follows the paper draft:
- task DAG edges encode workflow precedence;
- model DAG edges encode useful/valid transitions, sharing, warm starts, or reuse;
- the compatibility matrix couples task nodes with candidate models.
"""

from __future__ import annotations

import numpy as np

from dag_a2c.structures import DualDAGInstance, ModelSpec, RequestSpec, ServerSpec, TaskSpec, llm_transition_components, model_state_reuse_graph, model_state_reuse_tensor


def random_dag(num_nodes: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    graph = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src in range(num_nodes):
        for dst in range(src + 1, num_nodes):
            if rng.random() < edge_prob:
                graph[src, dst] = 1.0
    if num_nodes > 1 and not graph.any():
        graph[0, num_nodes - 1] = 1.0
    return graph


def _longest_path_depths(task_graph: np.ndarray) -> np.ndarray:
    depths = np.zeros(task_graph.shape[0], dtype=np.float32)
    for node in range(task_graph.shape[0] - 1, -1, -1):
        successors = np.where(task_graph[node] > 0)[0]
        if len(successors):
            depths[node] = 1.0 + float(np.max(depths[successors]))
    return depths


def generate_dual_dag_instance(
    num_tasks: int = 12,
    num_models: int = 6,
    num_servers: int = 3,
    num_task_types: int = 4,
    num_price_bins: int = 7,
    task_edge_prob: float = 0.22,
    model_edge_prob: float = 0.30,
    seed: int | None = None,
) -> DualDAGInstance:
    rng = np.random.default_rng(seed)
    task_graph = random_dag(num_tasks, task_edge_prob, rng)
    tasks: list[TaskSpec] = []
    for task_id in range(num_tasks):
        tasks.append(
            TaskSpec(
                task_id=task_id,
                task_type=int(rng.integers(0, num_task_types)),
                compute_demand=float(rng.uniform(1.0, 5.5)),
                input_size=float(rng.uniform(0.2, 2.0)),
                output_size=float(rng.uniform(0.1, 1.4)),
                min_quality=float(rng.uniform(0.58, 0.82)),
                kv_cache_size=float(rng.uniform(0.2, 1.8)),
                context_state_size=float(rng.uniform(0.15, 1.5)),
                tokenizer_cost=float(rng.uniform(0.04, 0.22)),
                prefix_overlap=float(rng.uniform(0.0, 0.8)),
                context_reuse_ratio=float(rng.uniform(0.2, 0.9)),
            )
        )

    models: list[ModelSpec] = []
    tokenizer_groups = max(2, min(4, num_models // 2))
    backbone_groups = max(2, min(4, num_models // 2))
    for model_id in range(num_models):
        rank = (model_id + 1) / max(1, num_models)
        supported = sorted(set(rng.choice(num_task_types, size=rng.integers(1, num_task_types + 1), replace=False)))
        backbone_id = int(model_id % backbone_groups)
        tokenizer_group = int((model_id // 2) % tokenizer_groups)
        models.append(
            ModelSpec(
                model_id=model_id,
                quality=float(np.clip(rng.normal(0.58 + 0.36 * rank, 0.04), 0.50, 0.99)),
                unit_cost=float(rng.uniform(0.18, 0.65) + 0.95 * rank),
                base_latency=float(rng.uniform(0.55, 1.70) + 0.45 * rank),
                memory=float(rng.uniform(1.2, 4.0) + 1.2 * rank),
                weight_size=float(rng.uniform(1.0, 5.0) + 2.5 * rank),
                supported_types=tuple(int(x) for x in supported),
                model_type=int(rng.integers(0, num_task_types)),
                backbone_id=backbone_id,
                tokenizer_group=tokenizer_group,
                context_format_group=tokenizer_group,
                adapter_group=int(rng.integers(0, 6)),
                supports_kv_cache=bool(rng.random() > 0.18),
                adapter_size=float(rng.uniform(0.08, 0.55) + 0.05 * rank),
            )
        )
    model_graph = model_state_reuse_graph(tuple(models))

    compatibility = np.zeros((num_tasks, num_models), dtype=np.float32)
    complexity = np.ones((num_tasks, num_models), dtype=np.float32)
    for task in tasks:
        feasible: list[int] = []
        for model in models:
            if task.task_type in model.supported_types and model.quality >= task.min_quality:
                compatibility[task.task_id, model.model_id] = 1.0
                feasible.append(model.model_id)
            complexity[task.task_id, model.model_id] = float(rng.uniform(0.75, 1.35))
        if not feasible:
            best = max(
                range(num_models),
                key=lambda idx: models[idx].quality - 0.15 * models[idx].base_latency,
            )
            compatibility[task.task_id, best] = 1.0

    servers: list[ServerSpec] = []
    for server_id in range(num_servers):
        servers.append(
            ServerSpec(
                server_id=server_id,
                compute_capacity=float(rng.uniform(4.0, 9.0)),
                memory_capacity=float(rng.uniform(8.0, 14.0)),
                bandwidth=float(rng.uniform(4.0, 12.0)),
                latency_factor=float(rng.uniform(0.75, 1.35)),
            )
        )

    switch_cost = np.zeros((num_models, num_models), dtype=np.float32)
    ref_task = max(tasks, key=lambda t: t.compute_demand)
    ref_server = max(servers, key=lambda s: s.bandwidth)
    for src in range(num_models):
        for dst in range(num_models):
            if src == dst:
                switch_cost[src, dst] = 0.0
            else:
                comps = llm_transition_components(models[src], models[dst], ref_task, ref_server, resident=True)
                switch_cost[src, dst] = float(comps["tokenizer"] + comps["kv"] + comps["adapter"] + comps["context"])

    total_compute = sum(task.compute_demand for task in tasks)
    critical_depth = float(np.max(_longest_path_depths(task_graph)) + 1.0)
    deadline = float(total_compute * rng.uniform(0.65, 0.95) / max(1.0, num_servers) + critical_depth * 2.0)
    valuation = float(total_compute * rng.uniform(1.25, 2.10))
    budget = float(valuation * rng.uniform(0.62, 0.95))
    price_bins = np.linspace(max(0.5, 0.25 * budget), 1.15 * budget, num_price_bins, dtype=np.float32)

    request = RequestSpec(
        task_graph=task_graph,
        tasks=tuple(tasks),
        budget=budget,
        deadline=deadline,
        valuation=valuation,
        alpha_quality=float(rng.uniform(1.6, 2.8)),
        beta_latency=float(rng.uniform(0.08, 0.22)),
        state_size=float(rng.uniform(0.4, 2.0)),
    )
    return DualDAGInstance(
        request=request,
        model_graph=model_graph,
        switch_cost=switch_cost,
        compatibility=compatibility,
        task_model_complexity=complexity,
        models=tuple(models),
        servers=tuple(servers),
        price_bins=price_bins,
        model_reuse=model_state_reuse_tensor(tuple(models)),
        transition_allowed=np.ones((len(models), len(models)), dtype=np.float32),
    )
