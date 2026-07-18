"""Benchmark-inspired request traces for dual-DAG tests.

The templates map public benchmark task categories into AIaaS workflows. They do
not download benchmark samples; instead they preserve the realistic workflow
shape, token scale, quality demand, and model specialization needed for
scheduling experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dag_a2c.structures import DualDAGInstance, ModelSpec, RequestSpec, ServerSpec, TaskSpec, llm_transition_components, model_state_reuse_graph, model_state_reuse_tensor


EMBED_RETRIEVE = 0
RERANK_CLASSIFY = 1
TEXT_GENERATE = 2
CODE_GENERATE = 3
VISION_LANGUAGE = 4


@dataclass(frozen=True)
class WorkflowTemplate:
    name: str
    source: str
    task_names: tuple[str, ...]
    task_types: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    compute: tuple[float, ...]
    io_scale: tuple[float, ...]
    min_quality: tuple[float, ...]
    deadline_base: float
    valuation_base: float
    state_size: float


TEMPLATES: dict[str, WorkflowTemplate] = {
    "mlperf_openorca_generation": WorkflowTemplate(
        name="mlperf_openorca_generation",
        source="MLPerf Inference Llama2-70B / OpenOrca-style text generation",
        task_names=("prompt_normalize", "safety_prefilter", "llm_generate", "safety_postfilter"),
        task_types=(RERANK_CLASSIFY, RERANK_CLASSIFY, TEXT_GENERATE, RERANK_CLASSIFY),
        edges=((0, 1), (1, 2), (2, 3)),
        compute=(0.8, 1.0, 8.8, 1.1),
        io_scale=(0.8, 0.5, 4.2, 0.7),
        min_quality=(0.62, 0.70, 0.86, 0.74),
        deadline_base=14.0,
        valuation_base=30.0,
        state_size=1.5,
    ),
    "longbench_rag_qa": WorkflowTemplate(
        name="longbench_rag_qa",
        source="LongBench multi-doc QA / retrieval-heavy RAG",
        task_names=("query_embedding", "vector_retrieval", "rerank_context", "long_context_reason", "answer_generate", "safety_check"),
        task_types=(EMBED_RETRIEVE, EMBED_RETRIEVE, RERANK_CLASSIFY, TEXT_GENERATE, TEXT_GENERATE, RERANK_CLASSIFY),
        edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
        compute=(1.0, 1.2, 2.0, 6.2, 4.0, 1.0),
        io_scale=(0.9, 2.6, 2.2, 4.8, 2.0, 0.7),
        min_quality=(0.62, 0.60, 0.72, 0.84, 0.82, 0.72),
        deadline_base=18.0,
        valuation_base=36.0,
        state_size=2.6,
    ),
    "longbench_summarization": WorkflowTemplate(
        name="longbench_summarization",
        source="LongBench GovReport/QMSum/MultiNews-style summarization",
        task_names=("document_chunk", "salience_extract", "draft_summary", "summary_refine", "safety_check"),
        task_types=(EMBED_RETRIEVE, RERANK_CLASSIFY, TEXT_GENERATE, TEXT_GENERATE, RERANK_CLASSIFY),
        edges=((0, 1), (1, 2), (1, 3), (2, 4), (3, 4)),
        compute=(1.6, 2.8, 6.4, 5.6, 1.0),
        io_scale=(5.5, 4.6, 3.8, 3.0, 0.7),
        min_quality=(0.60, 0.73, 0.84, 0.86, 0.72),
        deadline_base=20.0,
        valuation_base=38.0,
        state_size=3.4,
    ),
    "longbench_code_assistant": WorkflowTemplate(
        name="longbench_code_assistant",
        source="LongBench LCC/RepoBench-P-style code completion and repair",
        task_names=("repo_retrieval", "code_completion", "test_generation", "repair_pass", "final_safety"),
        task_types=(EMBED_RETRIEVE, CODE_GENERATE, CODE_GENERATE, CODE_GENERATE, RERANK_CLASSIFY),
        edges=((0, 1), (1, 2), (1, 3), (2, 3), (3, 4)),
        compute=(1.8, 6.8, 3.8, 5.2, 1.0),
        io_scale=(3.2, 2.6, 1.8, 2.0, 0.5),
        min_quality=(0.62, 0.84, 0.80, 0.86, 0.72),
        deadline_base=19.0,
        valuation_base=40.0,
        state_size=2.2,
    ),
}


def generate_trace_instance(
    template_name: str,
    seed: int | None = None,
    deadline_tightness: float = 1.0,
    budget_scale: float = 1.0,
    num_price_bins: int = 7,
) -> DualDAGInstance:
    rng = np.random.default_rng(seed)
    template = TEMPLATES[template_name]
    task_graph = np.zeros((len(template.task_names), len(template.task_names)), dtype=np.float32)
    for src, dst in template.edges:
        task_graph[src, dst] = 1.0

    tasks: list[TaskSpec] = []
    for task_id, (task_type, compute, io, min_quality) in enumerate(
        zip(template.task_types, template.compute, template.io_scale, template.min_quality)
    ):
        jitter = float(rng.lognormal(mean=0.0, sigma=0.08))
        tasks.append(
            TaskSpec(
                task_id=task_id,
                task_type=task_type,
                compute_demand=float(compute * jitter),
                input_size=float(io * rng.uniform(0.75, 1.25)),
                output_size=float(max(0.2, io * rng.uniform(0.20, 0.55))),
                min_quality=float(np.clip(min_quality + rng.normal(0.0, 0.015), 0.55, 0.92)),
                kv_cache_size=float(io * rng.uniform(0.55, 1.10) if task_type in (TEXT_GENERATE, CODE_GENERATE) else io * rng.uniform(0.12, 0.35)),
                context_state_size=float(template.state_size * rng.uniform(0.25, 0.70) + io * rng.uniform(0.05, 0.18)),
                tokenizer_cost=float(rng.uniform(0.05, 0.18) + 0.018 * io),
                prefix_overlap=float(rng.uniform(0.45, 0.85) if task_id > 0 and task_type in (TEXT_GENERATE, CODE_GENERATE) else 0.0),
                context_reuse_ratio=float(rng.uniform(0.55, 0.90) if task_id > 0 else 0.0),
            )
        )

    models = _realistic_model_pool()
    model_graph = _model_graph(models)
    compatibility = np.zeros((len(tasks), len(models)), dtype=np.float32)
    complexity = np.ones((len(tasks), len(models)), dtype=np.float32)
    for task in tasks:
        feasible = []
        for model in models:
            if task.task_type in model.supported_types and model.quality >= task.min_quality:
                compatibility[task.task_id, model.model_id] = 1.0
                feasible.append(model.model_id)
            complexity[task.task_id, model.model_id] = _complexity(task.task_type, model.model_id)
        if not feasible:
            fallback = max(range(len(models)), key=lambda idx: models[idx].quality if task.task_type in models[idx].supported_types else -1.0)
            compatibility[task.task_id, fallback] = 1.0

    servers = (
        ServerSpec(0, compute_capacity=10.5, memory_capacity=18.0, bandwidth=14.0, latency_factor=0.78, operating_cost_per_time=0.42),
        ServerSpec(1, compute_capacity=7.0, memory_capacity=13.5, bandwidth=9.0, latency_factor=1.00, operating_cost_per_time=0.30),
        ServerSpec(2, compute_capacity=4.8, memory_capacity=8.0, bandwidth=5.5, latency_factor=1.28, operating_cost_per_time=0.21),
    )
    switch_cost = _switch_cost(model_graph, models, tasks, servers)

    valuation = float(template.valuation_base * rng.uniform(0.90, 1.12))
    budget = float(valuation * budget_scale * rng.uniform(0.68, 0.95))
    deadline = float(template.deadline_base * deadline_tightness * rng.uniform(0.92, 1.10))
    price_bins = np.linspace(max(1.0, 0.30 * budget), max(1.2, 1.18 * budget), num_price_bins, dtype=np.float32)
    request = RequestSpec(
        task_graph=task_graph,
        tasks=tuple(tasks),
        budget=budget,
        deadline=deadline,
        valuation=valuation,
        alpha_quality=float(rng.uniform(2.0, 3.2)),
        beta_latency=float(rng.uniform(0.12, 0.26)),
        state_size=float(template.state_size * rng.uniform(0.85, 1.20)),
        service_fee=valuation,
        sla_penalty=float(0.35 * valuation),
    )
    return DualDAGInstance(
        request=request,
        model_graph=model_graph,
        switch_cost=switch_cost,
        compatibility=compatibility,
        task_model_complexity=complexity,
        models=models,
        servers=servers,
        price_bins=price_bins,
        model_reuse=model_state_reuse_tensor(models),
        transition_allowed=np.ones((len(models), len(models)), dtype=np.float32),
    )


def _realistic_model_pool() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(0, quality=0.66, unit_cost=0.18, base_latency=0.45, memory=1.2, weight_size=1.0, supported_types=(EMBED_RETRIEVE, RERANK_CLASSIFY), model_type=0, backbone_id=0, tokenizer_group=0, context_format_group=0, adapter_group=0, supports_kv_cache=False, adapter_size=0.12),
        ModelSpec(1, quality=0.77, unit_cost=0.34, base_latency=0.70, memory=2.0, weight_size=1.8, supported_types=(EMBED_RETRIEVE, RERANK_CLASSIFY, TEXT_GENERATE), model_type=1, backbone_id=1, tokenizer_group=1, context_format_group=1, adapter_group=0, supports_kv_cache=True, adapter_size=0.22),
        ModelSpec(2, quality=0.84, unit_cost=0.72, base_latency=1.20, memory=4.2, weight_size=4.0, supported_types=(TEXT_GENERATE, RERANK_CLASSIFY), model_type=2, backbone_id=1, tokenizer_group=1, context_format_group=1, adapter_group=1, supports_kv_cache=True, adapter_size=0.42),
        ModelSpec(3, quality=0.91, unit_cost=1.15, base_latency=1.85, memory=7.0, weight_size=7.8, supported_types=(TEXT_GENERATE, CODE_GENERATE), model_type=3, backbone_id=2, tokenizer_group=1, context_format_group=1, adapter_group=2, supports_kv_cache=True, adapter_size=0.72),
        ModelSpec(4, quality=0.88, unit_cost=0.95, base_latency=1.55, memory=5.8, weight_size=6.0, supported_types=(CODE_GENERATE, TEXT_GENERATE), model_type=3, backbone_id=2, tokenizer_group=2, context_format_group=2, adapter_group=3, supports_kv_cache=True, adapter_size=0.58),
        ModelSpec(5, quality=0.80, unit_cost=0.62, base_latency=1.05, memory=3.6, weight_size=3.2, supported_types=(VISION_LANGUAGE, TEXT_GENERATE, RERANK_CLASSIFY), model_type=4, backbone_id=1, tokenizer_group=1, context_format_group=1, adapter_group=4, supports_kv_cache=True, adapter_size=0.36),
        ModelSpec(6, quality=0.74, unit_cost=0.42, base_latency=0.80, memory=2.5, weight_size=2.2, supported_types=(CODE_GENERATE, RERANK_CLASSIFY), model_type=3, backbone_id=2, tokenizer_group=2, context_format_group=2, adapter_group=5, supports_kv_cache=True, adapter_size=0.26),
        ModelSpec(7, quality=0.70, unit_cost=0.24, base_latency=0.55, memory=1.8, weight_size=1.4, supported_types=(RERANK_CLASSIFY, EMBED_RETRIEVE), model_type=0, backbone_id=0, tokenizer_group=0, context_format_group=0, adapter_group=1, supports_kv_cache=False, adapter_size=0.16),
    )


def _model_graph(models: tuple[ModelSpec, ...]) -> np.ndarray:
    return model_state_reuse_graph(models)


def _complexity(task_type: int, model_id: int) -> float:
    table = {
        EMBED_RETRIEVE: {0: 0.72, 1: 0.90, 7: 0.80},
        RERANK_CLASSIFY: {0: 0.80, 1: 0.88, 2: 1.05, 5: 0.92, 6: 0.95, 7: 0.76},
        TEXT_GENERATE: {1: 1.15, 2: 0.92, 3: 1.18, 4: 1.06, 5: 1.12},
        CODE_GENERATE: {3: 1.05, 4: 0.90, 6: 1.18},
        VISION_LANGUAGE: {5: 0.95},
    }
    return float(table.get(task_type, {}).get(model_id, 1.35))


def _switch_cost(model_graph: np.ndarray, models: tuple[ModelSpec, ...], tasks: list[TaskSpec], servers: tuple[ServerSpec, ...]) -> np.ndarray:
    costs = np.zeros_like(model_graph, dtype=np.float32)
    ref_task = max(tasks, key=lambda t: t.compute_demand)
    ref_server = max(servers, key=lambda s: s.bandwidth)
    for src in range(len(models)):
        for dst in range(len(models)):
            if src == dst:
                costs[src, dst] = 0.0
            else:
                comps = llm_transition_components(models[src], models[dst], ref_task, ref_server, resident=True)
                costs[src, dst] = float(comps["tokenizer"] + comps["kv"] + comps["adapter"] + comps["context"])
    return costs
