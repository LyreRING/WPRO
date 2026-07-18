"""Core data structures for state-reuse-aware multi-LLM scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    task_type: int
    compute_demand: float
    input_size: float
    output_size: float
    min_quality: float
    kv_cache_size: float = 0.0
    context_state_size: float = 0.0
    tokenizer_cost: float = 0.0
    prefix_overlap: float = 0.0
    context_reuse_ratio: float = 0.0


@dataclass(frozen=True)
class ModelSpec:
    model_id: int
    quality: float
    unit_cost: float
    base_latency: float
    memory: float
    weight_size: float
    supported_types: tuple[int, ...]
    model_type: int = 0
    backbone_id: int = 0
    tokenizer_group: int = 0
    context_format_group: int = 0
    adapter_group: int = 0
    supports_kv_cache: bool = True
    adapter_size: float = 0.0


@dataclass(frozen=True)
class ServerSpec:
    server_id: int
    compute_capacity: float
    memory_capacity: float
    bandwidth: float
    latency_factor: float
    operating_cost_per_time: float = 0.25


@dataclass(frozen=True)
class RequestSpec:
    task_graph: np.ndarray
    tasks: tuple[TaskSpec, ...]
    budget: float
    deadline: float
    valuation: float
    alpha_quality: float
    beta_latency: float
    state_size: float
    # Posted service tariff and contractual SLA credit. The legacy economic
    # fields above remain only for backward compatibility with old scripts.
    service_fee: float = 0.0
    sla_penalty: float = 0.0


@dataclass(frozen=True)
class DualDAGInstance:
    request: RequestSpec
    # Historical name kept for compatibility. Semantically this is the
    # Model State Reuse Graph (MSRG): an edge m -> m' means LLM serving state
    # can be reused when transitioning from model m to model m'.
    model_graph: np.ndarray
    switch_cost: np.ndarray
    compatibility: np.ndarray
    task_model_complexity: np.ndarray
    models: tuple[ModelSpec, ...]
    servers: tuple[ServerSpec, ...]
    price_bins: np.ndarray
    # Static metadata-derived edge attributes ordered as
    # (tokenizer, KV/prefix, adapter/backbone, context compatibility).
    model_reuse: np.ndarray | None = None
    # Deployment/runtime transition permission. This is distinct from reuse:
    # a zero-reuse transition is normally executable after paying full cost.
    transition_allowed: np.ndarray | None = None

    @property
    def num_tasks(self) -> int:
        return len(self.request.tasks)

    @property
    def num_models(self) -> int:
        return len(self.models)

    @property
    def num_servers(self) -> int:
        return len(self.servers)


def model_state_reuse_graph(models: tuple[ModelSpec, ...]) -> np.ndarray:
    """Build binary MSRG support from metadata-derived edge attributes."""
    reuse = model_state_reuse_tensor(models)
    graph = (np.max(reuse, axis=-1) > 0.0).astype(np.float32)
    np.fill_diagonal(graph, 0.0)
    return graph


def model_state_reuse_tensor(models: tuple[ModelSpec, ...]) -> np.ndarray:
    """Return static MSRG edge vectors in [0, 1].

    The four channels are tokenizer/prompt compatibility, maximal KV-prefix
    compatibility, shared-backbone/adapter reuse, and context-format
    compatibility. Supported task types do not create MSRG edges; they form the
    separate task-model compatibility mask. Runtime prefix/context overlap and
    GPU residency are applied later for each workflow transition.
    """
    reuse = np.zeros((len(models), len(models), 4), dtype=np.float32)
    for src in models:
        for dst in models:
            same_tokenizer = src.tokenizer_group == dst.tokenizer_group
            same_backbone = src.backbone_id == dst.backbone_id
            rho_tok = 1.0 if same_tokenizer else 0.0
            rho_kv = 1.0 if same_tokenizer and same_backbone and src.supports_kv_cache and dst.supports_kv_cache else 0.0
            if src.model_id == dst.model_id:
                rho_adp = 1.0
            elif same_backbone:
                rho_adp = float(np.clip(1.0 - dst.adapter_size / max(dst.weight_size, 1e-6), 0.0, 1.0))
            else:
                rho_adp = 0.0
            rho_ctx = 1.0 if src.context_format_group == dst.context_format_group else 0.0
            reuse[src.model_id, dst.model_id] = (rho_tok, rho_kv, rho_adp, rho_ctx)
    return reuse


def model_state_reuse_vector(prev_model: ModelSpec, cur_model: ModelSpec, task: TaskSpec) -> tuple[float, float, float, float]:
    """Resolve a task-conditioned MSRG edge vector."""
    same_tokenizer = prev_model.tokenizer_group == cur_model.tokenizer_group
    same_backbone = prev_model.backbone_id == cur_model.backbone_id
    rho_tok = 1.0 if same_tokenizer else 0.0
    rho_kv_max = 1.0 if same_tokenizer and same_backbone and prev_model.supports_kv_cache and cur_model.supports_kv_cache else 0.0
    if prev_model.model_id == cur_model.model_id:
        rho_adp = 1.0
    elif same_backbone:
        rho_adp = float(np.clip(1.0 - cur_model.adapter_size / max(cur_model.weight_size, 1e-6), 0.0, 1.0))
    else:
        rho_adp = 0.0
    rho_ctx_format = 1.0 if prev_model.context_format_group == cur_model.context_format_group else 0.0
    rho_kv = rho_kv_max * float(np.clip(task.prefix_overlap, 0.0, 1.0))
    rho_ctx = float(np.clip(task.context_reuse_ratio, 0.0, 1.0)) * rho_ctx_format
    return rho_tok, rho_kv, rho_adp, rho_ctx


def llm_transition_components(
    prev_model: ModelSpec | None,
    cur_model: ModelSpec,
    task: TaskSpec,
    server: ServerSpec,
    resident: bool = False,
    has_predecessor_state: bool = True,
) -> dict[str, float]:
    """Return LLM-specific transition overhead components.

    The returned dictionary separates the cost into model loading, tokenizer /
    prompt preprocessing, KV-cache or prefill loss, adapter/backbone switching,
    and intermediate context-state transfer.
    """
    bandwidth = max(server.bandwidth, 1e-6)
    load = 0.0 if resident else cur_model.weight_size / bandwidth
    if prev_model is None:
        tokenizer = 0.0
        # Initial prefill belongs to profiled service time. This term measures
        # only extra re-prefill caused by losing predecessor KV state.
        kv = 0.0
        adapter = 0.0
        context = 0.0
    else:
        rho_tok, rho_kv, rho_adp, rho_ctx = model_state_reuse_vector(prev_model, cur_model, task)
        tokenizer = (1.0 - rho_tok) * task.tokenizer_cost
        kv = (1.0 - rho_kv) * task.kv_cache_size / bandwidth
        adapter = (1.0 - rho_adp) * cur_model.adapter_size / bandwidth
        context = (1.0 - rho_ctx) * task.context_state_size / bandwidth if has_predecessor_state else 0.0
    total = load + tokenizer + kv + adapter + context
    return {
        "load": float(load),
        "tokenizer": float(tokenizer),
        "kv": float(kv),
        "adapter": float(adapter),
        "context": float(context),
        "total": float(total),
    }
