"""WPR-A2C event-driven AIaaS workflow orchestration environment.

本文件是论文实验使用的核心 simulator。相比最初原型，这一版显式对齐论文
system model：
1. workflow 到达时立即做 admission decision，不再放入无限隐式等待队列；
2. stage 区分 llm/tool/communication 语义，只有 llm stage 进入 GPU ready queue；
3. GPU 状态机区分 resident / target / model_ready_time，cold load 不再被误认为已驻留；
4. 后继 stage ready 之前加入跨 server communication delay；
5. LLM 执行时间由 input tokens、output tokens、stage/model/GPU 和随机扰动共同决定；
6. ready_times 被真实维护，可统计 ready-to-start waiting latency。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np


ExecutionClass = Literal["llm", "tool", "communication"]

STAGE_TYPES = ("planner", "retriever", "reasoner", "generator", "verifier", "code-repair", "summarizer")
STAGE_INDEX = {name: idx for idx, name in enumerate(STAGE_TYPES)}
LLM_STAGE_TYPES = {"planner", "reasoner", "generator", "verifier", "code-repair", "summarizer"}


@dataclass(frozen=True)
class WPRStage:
    stage_id: int
    name: str
    stage_type: int
    work: float
    min_quality: float
    execution_class: ExecutionClass
    input_tokens_mean: float
    output_tokens_mean: float
    output_mb_mean: float
    tool_time_mean: float = 0.0


@dataclass(frozen=True)
class WPRTemplate:
    name: str
    stages: tuple[WPRStage, ...]
    edges: tuple[tuple[int, int], ...]
    deadline: float
    weight: float
    service_class: int


@dataclass(frozen=True)
class WPRModel:
    model_id: int
    name: str
    supported_types: tuple[int, ...]
    quality_by_type: tuple[float, ...]
    prefill_time_per_ktok: float
    decode_time_per_ktok: float
    memory: float
    weight_size: float
    backbone: int
    adapter_size: float


@dataclass(frozen=True)
class WPRGpu:
    gpu_id: int
    name: str
    speed: float
    memory: float
    bandwidth: float
    server_id: int


@dataclass
class WPRWorkflow:
    workflow_id: int
    template: WPRTemplate
    arrival: float
    admitted: bool
    completed: np.ndarray
    scheduled: np.ndarray
    finish_times: np.ndarray
    stage_model: np.ndarray
    stage_gpu: np.ndarray
    ready_times: np.ndarray
    start_times: np.ndarray
    input_tokens: np.ndarray
    expected_output_tokens: np.ndarray
    actual_output_tokens: np.ndarray
    output_mb: np.ndarray
    exec_jitter: np.ndarray


class WorkloadSource(Protocol):
    """统一 workload 输入接口。

    Synthetic 和 trace-driven workload 都返回已经实例化好的 WPRWorkflow 列表。
    环境后续的 admission、调度、通信和执行逻辑完全一致。
    """

    def load(self, env: "WPREnv") -> list[WPRWorkflow]:
        ...


@dataclass(frozen=True)
class SyntheticWorkloadSource:
    """参数化随机 workload，用于 controlled experiments。"""

    def load(self, env: "WPREnv") -> list[WPRWorkflow]:
        return env._generate_synthetic_arrivals()


@dataclass(frozen=True)
class TraceWorkloadSource:
    """CSV trace-driven workload loader。

    该 loader 使用真实 trace 中的 arrival timestamp 与 input/output token 长度，
    再将每条请求实例化为应用 workflow DAG。它不声称 trace 原生包含 DAG，而是实现：

        real request trace + application workflow template
        = trace-driven workflow instance
    """

    path: str | Path
    time_scale: float = 1.0
    max_requests: int | None = None
    start_time: float | None = None
    duration: float | None = None
    timestamp_col: str | None = None
    input_tokens_col: str | None = None
    output_tokens_col: str | None = None
    model_col: str | None = None
    elapsed_col: str | None = None
    deadline_mode: str = "template"
    deadline_multiplier: float = 2.5

    def load(self, env: "WPREnv") -> list[WPRWorkflow]:
        rows = self._read_rows()
        if not rows:
            return []
        timestamp_key = self._resolve_key(rows[0], self.timestamp_col, ("timestamp", "time", "arrival", "created_at"))
        input_key = self._resolve_key(rows[0], self.input_tokens_col, ("request tokens", "request_tokens", "input_tokens", "prompt_tokens", "input length"))
        output_key = self._resolve_key(rows[0], self.output_tokens_col, ("response tokens", "response_tokens", "output_tokens", "completion_tokens", "output length"))
        model_key = self._resolve_key(rows[0], self.model_col, ("model", "model_name", "engine", "type"), required=False)
        elapsed_key = self._resolve_key(rows[0], self.elapsed_col, ("elapsed time", "elapsed_time", "latency", "duration"), required=False)

        parsed: list[dict[str, Any]] = []
        for row in rows:
            ts = self._as_float(row.get(timestamp_key))
            if ts is None:
                continue
            parsed.append({"row": row, "timestamp": ts})
        if not parsed:
            return []
        parsed.sort(key=lambda x: x["timestamp"])
        t0 = parsed[0]["timestamp"] if self.start_time is None else float(self.start_time)

        workflows: list[WPRWorkflow] = []
        for item in parsed:
            raw_t = float(item["timestamp"])
            if raw_t < t0:
                continue
            arrival = (raw_t - t0) / max(self.time_scale, 1e-9)
            if self.duration is not None and arrival > self.duration:
                continue
            row = item["row"]
            request_tokens = self._as_float(row.get(input_key), default=1024.0)
            response_tokens = self._as_float(row.get(output_key), default=256.0)
            model_name = str(row.get(model_key, "")) if model_key else ""
            elapsed = self._as_float(row.get(elapsed_key), default=None) if elapsed_key else None
            template = env.select_trace_template(model_name, request_tokens, response_tokens)
            deadline_override = None
            if self.deadline_mode == "elapsed" and elapsed is not None:
                deadline_override = max(1.0, self.deadline_multiplier * elapsed / max(self.time_scale, 1e-9))
            elif self.deadline_mode == "relative":
                min_duration = env._estimate_min_template_duration(template)
                deadline_override = max(1.0, self.deadline_multiplier * min_duration)
            workflows.append(
                env.instantiate_workflow(
                    len(workflows),
                    template,
                    arrival,
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    deadline_override=deadline_override,
                )
            )
            if self.max_requests is not None and len(workflows) >= self.max_requests:
                break
        return workflows

    def _read_rows(self) -> list[dict[str, str]]:
        path = Path(self.path)
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _resolve_key(row: dict[str, str], explicit: str | None, candidates: tuple[str, ...], required: bool = True) -> str | None:
        if explicit:
            return explicit
        lookup = {k.strip().lower(): k for k in row}
        for cand in candidates:
            if cand.lower() in lookup:
                return lookup[cand.lower()]
        if required:
            raise ValueError(f"Cannot resolve trace column from candidates: {candidates}")
        return None

    @staticmethod
    def _as_float(value: Any, default: float | None = None) -> float | None:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


def build_workflow_templates() -> tuple[WPRTemplate, ...]:
    """设置主要 workflow 模式。

    注释里的重点：retriever/chunk/test 这类外部工具阶段不进入 GPU 调度队列；
    planner/reasoner/generator/verifier/code-repair/summarizer 才是 GPU-backed LLM
    stage。coding workflow 使用两类展开后的 DAG 模板表达 test 成功与失败修复分支。
    """

    S = STAGE_INDEX
    return (
        WPRTemplate(
            "deep_research",
            (
                WPRStage(0, "plan", S["planner"], 1.1, 0.72, "llm", 1.6, 0.35, 0.8),
                WPRStage(1, "web-search-a", S["retriever"], 1.0, 0.00, "tool", 0.3, 0.1, 4.5, 1.8),
                WPRStage(2, "web-search-b", S["retriever"], 1.0, 0.00, "tool", 0.3, 0.1, 4.2, 1.6),
                WPRStage(3, "reason", S["reasoner"], 5.3, 0.86, "llm", 5.6, 1.2, 1.6),
                WPRStage(4, "write", S["generator"], 4.1, 0.84, "llm", 4.4, 2.8, 2.4),
                WPRStage(5, "verify", S["verifier"], 1.5, 0.76, "llm", 3.2, 0.35, 0.6),
            ),
            ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)),
            deadline=34.0,
            weight=4.2,
            service_class=2,
        ),
        WPRTemplate(
            "rag_qa",
            (
                WPRStage(0, "query-plan", S["planner"], 0.8, 0.70, "llm", 0.9, 0.25, 0.4),
                WPRStage(1, "retrieve", S["retriever"], 1.0, 0.00, "tool", 0.2, 0.1, 3.5, 1.2),
                WPRStage(2, "reason", S["reasoner"], 4.4, 0.84, "llm", 4.0, 0.9, 1.2),
                WPRStage(3, "answer", S["generator"], 3.0, 0.82, "llm", 2.8, 1.4, 0.9),
                WPRStage(4, "check", S["verifier"], 1.0, 0.74, "llm", 1.8, 0.25, 0.4),
            ),
            ((0, 1), (1, 2), (2, 3), (3, 4)),
            deadline=26.0,
            weight=3.2,
            service_class=1,
        ),
        WPRTemplate(
            "coding_success",
            (
                WPRStage(0, "repo-context", S["retriever"], 1.0, 0.00, "tool", 0.3, 0.1, 5.0, 1.7),
                WPRStage(1, "generate-patch", S["generator"], 3.6, 0.82, "llm", 3.8, 1.8, 1.1),
                WPRStage(2, "run-tests", S["verifier"], 1.0, 0.00, "tool", 0.2, 0.1, 1.4, 2.2),
                WPRStage(3, "final-check", S["verifier"], 1.0, 0.76, "llm", 2.2, 0.35, 0.4),
            ),
            ((0, 1), (1, 2), (2, 3)),
            deadline=25.0,
            weight=3.6,
            service_class=2,
        ),
        WPRTemplate(
            "coding_repair",
            (
                WPRStage(0, "repo-context", S["retriever"], 1.0, 0.00, "tool", 0.3, 0.1, 5.0, 1.7),
                WPRStage(1, "generate-patch", S["generator"], 3.6, 0.82, "llm", 3.8, 1.8, 1.1),
                WPRStage(2, "run-tests", S["verifier"], 1.0, 0.00, "tool", 0.2, 0.1, 1.4, 2.2),
                WPRStage(3, "repair", S["code-repair"], 4.2, 0.86, "llm", 4.4, 1.5, 1.0),
                WPRStage(4, "retest", S["verifier"], 1.0, 0.00, "tool", 0.2, 0.1, 1.1, 1.8),
                WPRStage(5, "final-check", S["verifier"], 1.0, 0.76, "llm", 2.2, 0.35, 0.4),
            ),
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
            deadline=34.0,
            weight=4.1,
            service_class=2,
        ),
        WPRTemplate(
            "document_analysis",
            (
                WPRStage(0, "chunk", S["retriever"], 1.0, 0.00, "tool", 0.2, 0.1, 6.5, 1.5),
                WPRStage(1, "salience", S["summarizer"], 3.0, 0.80, "llm", 6.5, 1.2, 1.4),
                WPRStage(2, "draft", S["generator"], 3.7, 0.82, "llm", 4.8, 2.0, 1.8),
                WPRStage(3, "refine", S["summarizer"], 3.8, 0.84, "llm", 4.5, 1.6, 1.5),
                WPRStage(4, "verify", S["verifier"], 1.2, 0.74, "llm", 3.4, 0.35, 0.5),
            ),
            ((0, 1), (1, 2), (1, 3), (2, 4), (3, 4)),
            deadline=30.0,
            weight=3.5,
            service_class=1,
        ),
    )


def build_model_catalogue() -> tuple[WPRModel, ...]:
    S = STAGE_INDEX
    q = [0.0] * len(STAGE_TYPES)
    models: list[WPRModel] = []

    def add(
        name: str,
        types: list[str],
        qualities: list[float],
        prefill: float,
        decode: float,
        memory: float,
        weight_size: float,
        backbone: int,
        adapter: float,
    ) -> None:
        qq = q.copy()
        for t, val in zip(types, qualities):
            qq[S[t]] = val
        models.append(
            WPRModel(
                len(models),
                name,
                tuple(S[t] for t in types),
                tuple(qq),
                prefill,
                decode,
                memory,
                weight_size,
                backbone,
                adapter,
            )
        )

    add("Planner-S", ["planner", "verifier"], [0.78, 0.77], 0.18, 0.42, 1.8, 1.6, 0, 0.12)
    add("Retriever-E", ["summarizer"], [0.78], 0.16, 0.34, 2.0, 1.8, 0, 0.18)
    add("Reasoner-M", ["planner", "reasoner", "generator", "summarizer"], [0.82, 0.86, 0.83, 0.82], 0.34, 0.82, 4.8, 4.6, 1, 0.35)
    add("Reasoner-L", ["reasoner", "generator", "summarizer"], [0.92, 0.90, 0.89], 0.52, 1.15, 7.4, 7.8, 1, 0.70)
    add("Code-L", ["generator", "code-repair", "verifier"], [0.86, 0.91, 0.80], 0.42, 0.95, 6.5, 6.6, 2, 0.62)
    add("Safety-V", ["verifier", "planner"], [0.86, 0.75], 0.20, 0.46, 2.2, 2.1, 0, 0.15)
    return tuple(models)


def build_gpu_pool() -> tuple[WPRGpu, ...]:
    return (
        WPRGpu(0, "A100-80G", speed=1.28, memory=9.2, bandwidth=9.0, server_id=0),
        WPRGpu(1, "A100-40G", speed=1.00, memory=7.0, bandwidth=7.0, server_id=0),
        WPRGpu(2, "L40S", speed=0.78, memory=5.2, bandwidth=5.5, server_id=1),
    )


class WPREnv:
    """事件驱动 sequential assignment 环境。"""

    def __init__(
        self,
        horizon: float = 60.0,
        arrival_rate: float = 0.28,
        max_active: int = 7,
        admission_buffer: int = 0,
        seed: int = 0,
        demand_window: float = 10.0,
        drop_penalty: float = 1.0,
        network_bandwidth: float = 12.0,
        network_latency: float = 0.08,
        shaping_beta: float = 0.035,
        shaping_epsilon: float = 0.5,
        enable_potential_shaping: bool = True,
        workload_source: WorkloadSource | None = None,
    ) -> None:
        self.horizon = float(horizon)
        self.arrival_rate = float(arrival_rate)
        self.max_active = int(max_active)
        self.admission_buffer = int(admission_buffer)
        self.seed = int(seed)
        self.demand_window = float(demand_window)
        self.drop_penalty = float(drop_penalty)
        self.network_bandwidth = float(network_bandwidth)
        self.network_latency = float(network_latency)
        self.shaping_beta = float(shaping_beta)
        self.shaping_epsilon = float(shaping_epsilon)
        self.enable_potential_shaping = bool(enable_potential_shaping)
        self.workload_source = workload_source or SyntheticWorkloadSource()
        self.templates = build_workflow_templates()
        self.models = build_model_catalogue()
        self.gpus = build_gpu_pool()
        self.num_models = len(self.models)
        self.num_gpus = len(self.gpus)
        self.num_stage_types = len(STAGE_TYPES)
        self.max_stages = max(len(t.stages) for t in self.templates)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.pending = self.workload_source.load(self)
        self.all_workflows = list(self.pending)
        self.active: list[WPRWorkflow] = []
        self.done_workflows: list[WPRWorkflow] = []
        self.rejected_workflows: list[WPRWorkflow] = []
        self.dropped_workflows: list[WPRWorkflow] = []
        self.running: list[dict[str, float | int | str]] = []
        self.time = 0.0
        self.last_dt = 0.0
        self.step_count = 0
        self.gpu_available = np.zeros(self.num_gpus, dtype=np.float32)
        self.resident_model = np.full(self.num_gpus, -1, dtype=np.int64)
        self.target_model = np.full(self.num_gpus, -1, dtype=np.int64)
        self.model_ready_time = np.zeros(self.num_gpus, dtype=np.float32)
        self.gpu_state = np.asarray(["IDLE_RESIDENT"] * self.num_gpus, dtype=object)
        self.completed_value = 0.0
        self.sla_success = 0
        self.total_completed = 0
        self.admitted_count = 0
        self.history: list[dict[str, Any]] = []
        self._process_arrivals()
        self._advance_until_decision()
        return self.observe()

    @property
    def done(self) -> bool:
        no_work = not self.pending and not self.active and not self.running
        return bool(no_work or self.time >= self.horizon * 1.8 or self.step_count > 800)

    def idle_gpus(self) -> list[int]:
        return [g for g in range(self.num_gpus) if self.gpu_state[g] == "IDLE_RESIDENT" and self.gpu_available[g] <= self.time + 1e-9]

    def ready_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for slot, wf in enumerate(self.active):
            for sid in self.ready_stages(wf):
                pairs.append((slot, sid))
        return pairs

    def ready_stages(self, wf: WPRWorkflow) -> list[int]:
        ready: list[int] = []
        pred_map = self._pred_map(wf.template)
        for stage in wf.template.stages:
            i = stage.stage_id
            if wf.completed[i] or wf.scheduled[i] or stage.execution_class != "llm":
                continue
            if wf.ready_times[i] <= self.time + 1e-9 and all(wf.completed[p] for p in pred_map[i]):
                ready.append(i)
        return ready

    def feasible_actions_for_gpu(self, gpu_id: int, used_pairs: set[tuple[int, int]] | None = None) -> list[tuple[int, int, int, int]]:
        used_pairs = used_pairs or set()
        out: list[tuple[int, int, int, int]] = []
        gpu = self.gpus[gpu_id]
        for slot, sid in self.ready_pairs():
            if (slot, sid) in used_pairs:
                continue
            stage = self.active[slot].template.stages[sid]
            for model in self.models:
                if self.model_feasible(stage, model) and model.memory <= gpu.memory + 1e-9:
                    out.append((slot, sid, model.model_id, gpu_id))
        return out

    def all_feasible_actions(self) -> list[tuple[int, int, int, int]]:
        actions: list[tuple[int, int, int, int]] = []
        for g in self.idle_gpus():
            actions.extend(self.feasible_actions_for_gpu(g))
        return actions

    def has_future_external_event(self) -> bool:
        if self.pending and self.pending[0].arrival > self.time + 1e-9:
            return True
        return any(float(x["finish"]) > self.time + 1e-9 or float(x.get("prep_done", np.inf)) > self.time + 1e-9 for x in self.running)

    def step(self, assignments: list[tuple[int, int, int, int]]) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """执行同一 orchestration event 的 assignment set，然后推进到下一决策事件。"""

        self.step_count += 1
        before = self.time
        signature_before = self.state_signature()
        potential_before = self.potential()
        used_pairs: set[tuple[int, int]] = set()
        used_gpus: set[int] = set()
        for slot, sid, mid, gid in assignments:
            if slot < 0 or sid < 0 or mid < 0:
                continue
            if gid in used_gpus or (slot, sid) in used_pairs or gid not in self.idle_gpus():
                continue
            valid = (slot, sid, mid, gid) in self.feasible_actions_for_gpu(gid, used_pairs)
            if not valid:
                continue
            self._schedule_llm(slot, sid, mid, gid)
            used_pairs.add((slot, sid))
            used_gpus.add(gid)

        value_before = self.completed_value
        success_before = self.sla_success
        dropped_before = len(self.dropped_workflows)
        rejected_before = len(self.rejected_workflows)
        if used_pairs:
            self._advance_until_decision()
        else:
            self._advance_to_next_external_event()
            self._advance_until_decision()
        self.last_dt = max(0.0, self.time - before)
        signature_after = self.state_signature()
        if self.last_dt <= 1e-12 and signature_after == signature_before:
            raise RuntimeError("Zero-time transition without state change detected; event boundary is inconsistent.")
        terminal_reward = (
            self.completed_value
            - value_before
            - self.drop_penalty * (len(self.dropped_workflows) - dropped_before)
            - self.drop_penalty * (len(self.rejected_workflows) - rejected_before)
        )
        discount = float(np.exp(-self.shaping_beta * self.last_dt))
        shaping_reward = discount * self.potential() - potential_before if self.enable_potential_shaping else 0.0
        reward = terminal_reward + shaping_reward
        info = {
            "dt": self.last_dt,
            "new_success": self.sla_success - success_before,
            "assignments": len(assignments),
            "valid_assignments": len(used_pairs),
            "terminal_reward": float(terminal_reward),
            "shaping_reward": float(shaping_reward),
        }
        return self.observe(), float(reward), self.done, info

    def model_feasible(self, stage: WPRStage, model: WPRModel) -> bool:
        return stage.execution_class == "llm" and stage.stage_type in model.supported_types and model.quality_by_type[stage.stage_type] >= stage.min_quality

    def prep_time(self, model_id: int, gpu_id: int) -> float:
        """模型准备时间：resident hit / adapter transition / cold load。"""

        current = int(self.resident_model[gpu_id])
        target = self.models[model_id]
        if current == model_id and self.gpu_state[gpu_id] == "IDLE_RESIDENT":
            return 0.0
        gpu = self.gpus[gpu_id]
        if current >= 0 and self.models[current].backbone == target.backbone:
            return float(target.adapter_size / gpu.bandwidth + 0.08)
        return float(target.weight_size / gpu.bandwidth + 0.22)

    def exec_time(self, slot: int, stage_id: int, model_id: int, gpu_id: int) -> float:
        """Backward-compatible deterministic expected execution time."""

        return self.expected_exec_time(slot, stage_id, model_id, gpu_id)

    def expected_exec_time(self, slot: int, stage_id: int, model_id: int, gpu_id: int) -> float:
        """Deterministic expected LLM execution time for scoring and baselines."""

        wf = self.active[slot]
        stage = wf.template.stages[stage_id]
        model = self.models[model_id]
        gpu = self.gpus[gpu_id]
        prefill = model.prefill_time_per_ktok * (wf.input_tokens[stage_id] / 1000.0) / gpu.speed
        decode = model.decode_time_per_ktok * (wf.actual_output_tokens[stage_id] / 1000.0) / gpu.speed
        semantic_work = 0.18 * stage.work / gpu.speed
        return float(prefill + decode + semantic_work)

    def sample_exec_time(self, slot: int, stage_id: int, model_id: int, gpu_id: int) -> float:
        """Actual execution time from the pre-sampled counterfactual trace table."""

        expected = self.expected_exec_time(slot, stage_id, model_id, gpu_id)
        jitter = float(self.active[slot].exec_jitter[stage_id, model_id, gpu_id])
        return float(expected * jitter)

    def tool_time(self, wf: WPRWorkflow, stage_id: int) -> float:
        stage = wf.template.stages[stage_id]
        noise = float(self.rng.lognormal(mean=0.0, sigma=0.15))
        return float(max(0.05, stage.tool_time_mean * noise + 0.03 * stage.work))

    def communication_delay(self, wf: WPRWorkflow, pred_id: int, succ_id: int) -> float:
        pred_gpu = int(wf.stage_gpu[pred_id])
        if pred_gpu < 0:
            pred_server = 2
            bw = self.network_bandwidth * 0.8
        else:
            pred_server = self.gpus[pred_gpu].server_id
            bw = min(self.network_bandwidth, self.gpus[pred_gpu].bandwidth)
        # successor 尚未分配 GPU，只能用平台平均网络估计 input availability。
        cross_server = 1.0 if pred_server != 0 else 0.35
        return float(self.network_latency * cross_server + wf.output_mb[pred_id] / max(1e-6, bw))

    def input_transfer_delay(self, wf: WPRWorkflow, stage_id: int, target_gpu_id: int) -> float:
        """Dispatch-time input transfer cost coupled with the selected target GPU.

        A successor can be dependency-ready, but its actual execution starts after
        its inputs are transferred to the chosen GPU/server. This avoids computing
        placement-dependent communication before the placement decision exists.
        """

        pred_map = self._pred_map(wf.template)
        if not pred_map[stage_id]:
            return 0.0
        target_server = self.gpus[target_gpu_id].server_id
        target_bw = self.gpus[target_gpu_id].bandwidth
        delays = []
        for pred in pred_map[stage_id]:
            pred_gpu = int(wf.stage_gpu[pred])
            if pred_gpu >= 0:
                pred_server = self.gpus[pred_gpu].server_id
                bw = min(target_bw, self.gpus[pred_gpu].bandwidth, self.network_bandwidth)
            else:
                pred_server = 2
                bw = min(target_bw, self.network_bandwidth * 0.8)
            latency = self.network_latency * (1.0 if pred_server != target_server else 0.25)
            delays.append(latency + wf.output_mb[pred] / max(1e-6, bw))
        return float(max(delays, default=0.0))

    def potential(self) -> float:
        """Potential-based reward shaping term Phi(S).

        Phi(S) rewards useful workflow progress under deadline pressure. The
        environment still optimizes terminal SLA-compliant weighted value; this
        term only improves credit assignment through gamma*Phi(S')-Phi(S).
        """

        total = 0.0
        for wf in self.active:
            progress = float(np.mean(wf.completed))
            cp0 = max(1e-6, sum((st.work if st.execution_class == "llm" else st.tool_time_mean) for st in wf.template.stages))
            cp_done = max(0.0, 1.0 - self.remaining_critical_path(wf) / cp0)
            alpha = 0.5 * progress + 0.5 * cp_done
            slack = wf.arrival + wf.template.deadline - self.time
            slack_ratio = float(np.clip(slack / max(wf.template.deadline, self.shaping_epsilon), 0.0, 1.0))
            total += wf.template.weight * alpha * slack_ratio
        return float(total)

    def state_signature(self) -> tuple:
        """Compact discrete signature used to guard against zero-time no-op loops."""

        active_sig = tuple(
            (
                wf.workflow_id,
                tuple(bool(x) for x in wf.completed),
                tuple(bool(x) for x in wf.scheduled),
                tuple(float(np.round(x, 6)) if np.isfinite(x) else "inf" for x in wf.ready_times),
            )
            for wf in self.active
        )
        running_sig = tuple(
            sorted(
                (
                    int(x["workflow_id"]),
                    int(x["stage_id"]),
                    int(x["gpu_id"]),
                    str(x["kind"]),
                    float(np.round(float(x["finish"]), 6)),
                    float(np.round(float(x.get("prep_done", 0.0)), 6)),
                )
                for x in self.running
            )
        )
        return (
            float(np.round(self.time, 6)),
            len(self.pending),
            active_sig,
            running_sig,
            tuple(int(x) for x in self.resident_model),
            tuple(int(x) for x in self.target_model),
            tuple(str(x) for x in self.gpu_state),
            len(self.done_workflows),
            len(self.rejected_workflows),
            len(self.dropped_workflows),
        )

    def observe(self) -> dict[str, np.ndarray]:
        prep_frac = float(np.mean(self.gpu_state == "PREPARING"))
        run_frac = float(np.mean(self.gpu_state == "RUNNING"))
        return {
            "workflow_features": self.workflow_progress_features(),
            "residency": self.residency_features(),
            "global": np.asarray(
                [
                    self.time / max(self.horizon, 1e-6),
                    len(self.active) / self.max_active,
                    len(self.pending) / max(1, len(self.all_workflows)),
                    len(self.idle_gpus()) / self.num_gpus,
                    prep_frac,
                    run_frac,
                    len(self.rejected_workflows) / max(1, len(self.all_workflows)),
                ],
                dtype=np.float32,
            ),
        }

    def state_vector(self) -> np.ndarray:
        obs = self.observe()
        return np.concatenate([v.reshape(-1) for v in obs.values()]).astype(np.float32)

    def workflow_progress_features(self) -> np.ndarray:
        """每个 active workflow 的 progress/future-DAG 特征。"""

        feats = np.zeros((self.max_active, 9 + self.num_stage_types), dtype=np.float32)
        for slot, wf in enumerate(self.active[: self.max_active]):
            n = len(wf.template.stages)
            ready = self.ready_stages(wf)
            slack = wf.arrival + wf.template.deadline - self.time
            remaining_cp = self.remaining_critical_path(wf)
            future_types = np.zeros(self.num_stage_types, dtype=np.float32)
            for stage in wf.template.stages:
                if not wf.completed[stage.stage_id]:
                    future_types[stage.stage_type] += stage.work
            if np.sum(future_types) > 0:
                future_types /= np.sum(future_types)
            queue_wait = 0.0
            ready_waits = [max(0.0, self.time - float(wf.ready_times[sid])) for sid in ready]
            if ready_waits:
                queue_wait = float(np.mean(ready_waits))
            llm_left = sum(1 for st in wf.template.stages if st.execution_class == "llm" and not wf.completed[st.stage_id])
            tool_left = sum(1 for st in wf.template.stages if st.execution_class != "llm" and not wf.completed[st.stage_id])
            feats[slot] = np.concatenate(
                [
                    np.asarray(
                        [
                            1.0,
                            float(np.mean(wf.completed)),
                            len(ready) / max(1, n),
                            remaining_cp / max(wf.template.deadline, 1e-6),
                            np.clip(slack / max(wf.template.deadline, 1e-6), -1.0, 1.0),
                            wf.template.weight / 5.0,
                            queue_wait / max(wf.template.deadline, 1e-6),
                            llm_left / max(1, n),
                            tool_left / max(1, n),
                        ],
                        dtype=np.float32,
                    ),
                    future_types,
                ]
            )
        return feats

    def residency_features(self) -> np.ndarray:
        feats = np.zeros((self.num_gpus, 2 * self.num_models + 6), dtype=np.float32)
        for g in range(self.num_gpus):
            if self.resident_model[g] >= 0:
                feats[g, int(self.resident_model[g])] = 1.0
            if self.target_model[g] >= 0:
                feats[g, self.num_models + int(self.target_model[g])] = 1.0
            offset = 2 * self.num_models
            feats[g, offset:] = (
                self.gpus[g].speed / max(x.speed for x in self.gpus),
                self.gpus[g].memory / max(x.memory for x in self.gpus),
                self.gpus[g].bandwidth / max(x.bandwidth for x in self.gpus),
                float(self.gpu_state[g] == "IDLE_RESIDENT"),
                float(self.gpu_state[g] == "PREPARING"),
                float(self.gpu_state[g] == "RUNNING"),
            )
        return feats

    def oracle_dag_demand_target(self, window: float | None = None) -> np.ndarray:
        """DAG-oracle future demand target。

        它不是 rollout 真实未来标签，而是当前 unfinished DAG 在窗口 H 内可能释放的
        LLM 模型需求强度。论文里应称 oracle_dag_demand_target 或 auxiliary
        DAG demand label，避免写成不可观测 true future。
        """

        H = self.demand_window if window is None else float(window)
        demand = np.zeros(self.num_models, dtype=np.float32)
        for wf in self.active:
            est_release = self._earliest_unfinished_release_times(wf)
            slack = max(0.1, wf.arrival + wf.template.deadline - self.time)
            urgency = wf.template.weight / slack
            for stage in wf.template.stages:
                sid = stage.stage_id
                if stage.execution_class != "llm" or wf.completed[sid] or wf.scheduled[sid]:
                    continue
                if est_release[sid] - self.time <= H + 1e-9:
                    feasible_models = [m for m in self.models if self.model_feasible(stage, m)]
                    for model in feasible_models:
                        demand[model.model_id] += urgency * stage.work / max(1, len(feasible_models))
        if np.max(demand) > 0:
            demand = demand / np.max(demand)
        return demand.astype(np.float32)

    def true_future_model_demand(self, window: float | None = None) -> np.ndarray:
        return self.oracle_dag_demand_target(window)

    def remaining_critical_path(self, wf: WPRWorkflow) -> float:
        succ_map = self._succ_map(wf.template)
        memo: dict[int, float] = {}

        def cp(i: int) -> float:
            if wf.completed[i]:
                return 0.0
            if i in memo:
                return memo[i]
            stage = wf.template.stages[i]
            work = stage.work if stage.execution_class == "llm" else stage.tool_time_mean
            memo[i] = work + max((cp(s) for s in succ_map[i]), default=0.0)
            return memo[i]

        return max((cp(s.stage_id) for s in wf.template.stages if not wf.completed[s.stage_id]), default=0.0)

    def final_metrics(self) -> dict[str, float]:
        latencies = [float(np.nanmax(wf.finish_times) - wf.arrival) for wf in self.done_workflows]
        ready_waits: list[float] = []
        for wf in self.done_workflows:
            for st in wf.template.stages:
                if st.execution_class == "llm" and np.isfinite(wf.start_times[st.stage_id]):
                    ready_waits.append(max(0.0, float(wf.start_times[st.stage_id] - wf.ready_times[st.stage_id])))
        total = len(self.all_workflows)
        weighted_possible = sum(wf.template.weight for wf in self.all_workflows)
        episode_time = max(self.time, 1e-9)
        return {
            "weighted_completed_value": float(self.completed_value),
            "weighted_goodput": float(self.completed_value),
            "weighted_goodput_rate": float(self.completed_value / episode_time),
            "weighted_goodput_ratio": float(self.completed_value / max(1e-9, weighted_possible)),
            "sla_success_ratio": float(self.sla_success / max(1, total)),
            "completion_ratio": float(len(self.done_workflows) / max(1, total)),
            "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "avg_latency": float(np.mean(latencies)) if latencies else 0.0,
            "avg_ready_wait": float(np.mean(ready_waits)) if ready_waits else 0.0,
            "rejected": float(len(self.rejected_workflows)),
            "dropped": float(len(self.dropped_workflows)),
            "admitted": float(self.admitted_count),
            "arrived": float(total),
            "steps": float(self.step_count),
        }

    def _schedule_llm(self, slot: int, stage_id: int, model_id: int, gpu_id: int) -> None:
        wf = self.active[slot]
        input_delay = self.input_transfer_delay(wf, stage_id, gpu_id)
        prep = self.prep_time(model_id, gpu_id)
        exec_t = self.sample_exec_time(slot, stage_id, model_id, gpu_id)
        prep_done = self.time + input_delay + prep
        finish = prep_done + exec_t
        wf.scheduled[stage_id] = True
        wf.stage_model[stage_id] = model_id
        wf.stage_gpu[stage_id] = gpu_id
        wf.start_times[stage_id] = self.time + input_delay
        wf.finish_times[stage_id] = finish
        self.gpu_available[gpu_id] = finish
        self.target_model[gpu_id] = model_id
        self.model_ready_time[gpu_id] = prep_done
        self.gpu_state[gpu_id] = "PREPARING" if prep > 1e-9 else "RUNNING"
        self.running.append(
            {
                "finish": finish,
                "prep_done": prep_done,
                "workflow_id": wf.workflow_id,
                "stage_id": stage_id,
                "gpu_id": gpu_id,
                "model_id": model_id,
                "kind": "llm",
            }
        )
        self.history.append(
            {
                "time": self.time,
                "event": "dispatch",
                "workflow": wf.workflow_id,
                "template": wf.template.name,
                "stage": stage_id,
                "stage_type": STAGE_TYPES[wf.template.stages[stage_id].stage_type],
                "model": model_id,
                "gpu": gpu_id,
                "prep": prep,
                "input_transfer": input_delay,
                "exec": exec_t,
                "prep_done": prep_done,
                "finish": finish,
                "ready_wait": float(self.time - wf.ready_times[stage_id]),
            }
        )

    def _start_ready_tool_stages(self) -> bool:
        changed = False
        for wf in self.active:
            pred_map = self._pred_map(wf.template)
            for stage in wf.template.stages:
                sid = stage.stage_id
                if stage.execution_class == "llm" or wf.completed[sid] or wf.scheduled[sid]:
                    continue
                if wf.ready_times[sid] <= self.time + 1e-9 and all(wf.completed[p] for p in pred_map[sid]):
                    duration = self.tool_time(wf, sid)
                    finish = self.time + duration
                    wf.scheduled[sid] = True
                    wf.start_times[sid] = self.time
                    wf.stage_gpu[sid] = -1
                    wf.stage_model[sid] = -1
                    wf.finish_times[sid] = finish
                    self.running.append(
                        {
                            "finish": finish,
                            "prep_done": self.time,
                            "workflow_id": wf.workflow_id,
                            "stage_id": sid,
                            "gpu_id": -1,
                            "model_id": -1,
                            "kind": "tool",
                        }
                    )
                    self.history.append({"time": self.time, "event": "start_tool", "workflow": wf.workflow_id, "template": wf.template.name, "stage": sid, "finish": finish})
                    changed = True
        return changed

    def _advance_until_decision(self) -> None:
        while not self.done:
            self._drop_impossible_workflows()
            self._start_ready_tool_stages()
            if self.idle_gpus() and self.ready_pairs():
                return
            next_times = []
            if self.pending:
                next_times.append(self.pending[0].arrival)
            if self.running:
                next_times.extend(float(x["finish"]) for x in self.running)
                next_times.extend(float(x["prep_done"]) for x in self.running if float(x["prep_done"]) > self.time + 1e-9)
            future_times = [t for t in next_times if t > self.time + 1e-9]
            if not future_times:
                return
            self.time = float(min(future_times))
            self._complete_due()
            self._process_arrivals()

    def _advance_to_next_external_event(self) -> None:
        next_times = []
        if self.pending:
            next_times.append(self.pending[0].arrival)
        if self.running:
            next_times.extend(float(x["finish"]) for x in self.running)
            next_times.extend(float(x["prep_done"]) for x in self.running if float(x["prep_done"]) > self.time + 1e-9)
        future_times = [t for t in next_times if t > self.time + 1e-9]
        if future_times:
            self.time = float(min(future_times))
            self._complete_due()
            self._process_arrivals()

    def _complete_due(self) -> None:
        for item in self.running:
            gid = int(item["gpu_id"])
            if gid >= 0 and self.gpu_state[gid] == "PREPARING" and float(item["prep_done"]) <= self.time + 1e-9:
                self.resident_model[gid] = int(item["model_id"])
                self.gpu_state[gid] = "RUNNING"
                self.history.append({"time": self.time, "event": "model_ready", "gpu": gid, "model": int(item["model_id"])})

        due = [x for x in self.running if float(x["finish"]) <= self.time + 1e-9]
        self.running = [x for x in self.running if float(x["finish"]) > self.time + 1e-9]
        completed_ids: set[int] = set()
        for item in due:
            wid = int(item["workflow_id"])
            sid = int(item["stage_id"])
            gid = int(item["gpu_id"])
            for wf in self.active:
                if wf.workflow_id == wid:
                    wf.completed[sid] = True
                    self._update_successor_ready_times(wf, sid)
                    if np.all(wf.completed):
                        completed_ids.add(wid)
                    break
            if gid >= 0:
                self.resident_model[gid] = int(item["model_id"])
                self.target_model[gid] = -1
                self.gpu_available[gid] = self.time
                self.gpu_state[gid] = "IDLE_RESIDENT"

        if completed_ids:
            keep: list[WPRWorkflow] = []
            for wf in self.active:
                if wf.workflow_id in completed_ids:
                    self.done_workflows.append(wf)
                    self.total_completed += 1
                    latency = float(np.nanmax(wf.finish_times) - wf.arrival)
                    if latency <= wf.template.deadline + 1e-9:
                        self.sla_success += 1
                        self.completed_value += wf.template.weight
                    self.history.append({"time": self.time, "event": "complete_workflow", "workflow": wf.workflow_id, "template": wf.template.name, "latency": latency, "sla_success": latency <= wf.template.deadline + 1e-9, "weight": wf.template.weight})
                else:
                    keep.append(wf)
            self.active = keep

    def _update_successor_ready_times(self, wf: WPRWorkflow, completed_stage: int) -> None:
        pred_map = self._pred_map(wf.template)
        succ_map = self._succ_map(wf.template)
        for succ in succ_map[completed_stage]:
            if not all(wf.completed[p] for p in pred_map[succ]):
                continue
            # Dependency-ready time only requires predecessor outputs to exist.
            # Placement-dependent input transfer is charged later at dispatch.
            release = max(float(wf.finish_times[p]) for p in pred_map[succ])
            if not np.isfinite(wf.ready_times[succ]) or wf.ready_times[succ] < release:
                wf.ready_times[succ] = np.float32(release)

    def _process_arrivals(self) -> None:
        while self.pending and self.pending[0].arrival <= self.time + 1e-9:
            wf = self.pending.pop(0)
            can_admit = len(self.active) < self.max_active and self._sla_feasible_at_admission(wf)
            if can_admit:
                wf.admitted = True
                self.admitted_count += 1
                self._initialize_source_ready_times(wf)
                self.active.append(wf)
                self.history.append({"time": self.time, "event": "admit", "workflow": wf.workflow_id, "template": wf.template.name})
            else:
                self.rejected_workflows.append(wf)
                self.history.append({"time": self.time, "event": "reject", "workflow": wf.workflow_id, "template": wf.template.name})

    def _initialize_source_ready_times(self, wf: WPRWorkflow) -> None:
        pred_map = self._pred_map(wf.template)
        for st in wf.template.stages:
            if not pred_map[st.stage_id]:
                wf.ready_times[st.stage_id] = np.float32(max(self.time, wf.arrival))

    def _sla_feasible_at_admission(self, wf: WPRWorkflow) -> bool:
        estimate = self._estimate_min_template_duration(wf.template)
        queue = self._queue_workload_estimate()
        prep = self._prep_residency_estimate(wf.template)
        finish_est = self.time - wf.arrival + queue + estimate + prep
        return bool(finish_est <= wf.template.deadline)

    def _queue_workload_estimate(self) -> float:
        """Conservative workload-aware admission queue estimate."""

        unfinished = 0.0
        for wf in self.active:
            unfinished += max(0.0, self.remaining_critical_path(wf))
        running_left = sum(max(0.0, float(x["finish"]) - self.time) for x in self.running)
        capacity = max(1e-6, sum(g.speed for g in self.gpus))
        return float(0.35 * unfinished / capacity + 0.50 * running_left / max(1, self.num_gpus))

    def _prep_residency_estimate(self, template: WPRTemplate) -> float:
        prep = 0.0
        for stage in template.stages:
            if stage.execution_class != "llm":
                continue
            best = np.inf
            for model in self.models:
                if not self.model_feasible(stage, model):
                    continue
                for gpu in self.gpus:
                    if model.memory <= gpu.memory + 1e-9:
                        current = int(self.resident_model[gpu.gpu_id])
                        if current == model.model_id:
                            best = min(best, 0.0)
                        elif current >= 0 and self.models[current].backbone == model.backbone:
                            best = min(best, model.adapter_size / gpu.bandwidth + 0.08)
                        else:
                            best = min(best, model.weight_size / gpu.bandwidth + 0.22)
            prep += 0.25 * float(best if np.isfinite(best) else 0.5)
        return float(prep)

    def _estimate_min_template_duration(self, template: WPRTemplate) -> float:
        succ_map = self._succ_map(template)
        memo: dict[int, float] = {}

        def fastest_stage(stage: WPRStage) -> float:
            if stage.execution_class != "llm":
                return stage.tool_time_mean
            best = np.inf
            for model in self.models:
                if not self.model_feasible(stage, model):
                    continue
                for gpu in self.gpus:
                    if model.memory <= gpu.memory + 1e-9:
                        in_tok = stage.input_tokens_mean
                        out_tok = stage.output_tokens_mean
                        t = model.prefill_time_per_ktok * in_tok / 1000.0 / gpu.speed + model.decode_time_per_ktok * out_tok / 1000.0 / gpu.speed + 0.18 * stage.work / gpu.speed
                        best = min(best, t)
            return float(best if np.isfinite(best) else stage.work)

        def cp(i: int) -> float:
            if i in memo:
                return memo[i]
            stage = template.stages[i]
            memo[i] = fastest_stage(stage) + max((cp(s) for s in succ_map[i]), default=0.0)
            return memo[i]

        return max((cp(s.stage_id) for s in template.stages), default=0.0)

    def _drop_impossible_workflows(self) -> None:
        keep: list[WPRWorkflow] = []
        for wf in self.active:
            if self.time - wf.arrival > wf.template.deadline + 1.2 * self.remaining_critical_path(wf):
                self.dropped_workflows.append(wf)
                self.history.append({"time": self.time, "event": "drop", "workflow": wf.workflow_id, "template": wf.template.name})
            else:
                keep.append(wf)
        self.active = keep

    def instantiate_workflow(
        self,
        workflow_id: int,
        template: WPRTemplate,
        arrival: float,
        request_tokens: float | None = None,
        response_tokens: float | None = None,
        deadline_override: float | None = None,
    ) -> WPRWorkflow:
        """Instantiate a workflow DAG from either synthetic or trace inputs."""

        n = len(template.stages)
        if deadline_override is not None:
            template = WPRTemplate(
                template.name,
                template.stages,
                template.edges,
                deadline=float(deadline_override),
                weight=template.weight,
                service_class=template.service_class,
            )
        if request_tokens is None:
            input_tokens = np.asarray([max(64.0, self.rng.lognormal(np.log(st.input_tokens_mean * 1000.0), 0.22)) for st in template.stages], dtype=np.float32)
        else:
            input_tokens = self.trace_stage_input_tokens(template, float(request_tokens), float(response_tokens or max(16.0, request_tokens * 0.15)))
        if response_tokens is None:
            expected_out = np.asarray([max(16.0, st.output_tokens_mean * 1000.0) for st in template.stages], dtype=np.float32)
            actual_out = np.asarray([max(16.0, self.rng.lognormal(np.log(max(16.0, st.output_tokens_mean * 1000.0)), 0.25)) for st in template.stages], dtype=np.float32)
        else:
            actual_out = self.trace_stage_output_tokens(template, float(request_tokens or 1024.0), float(response_tokens))
            expected_out = actual_out.copy()
        output_mb = np.asarray([max(0.05, self.rng.lognormal(np.log(max(0.05, st.output_mb_mean)), 0.20)) for st in template.stages], dtype=np.float32)
        exec_jitter = self.rng.lognormal(mean=0.0, sigma=0.08, size=(n, self.num_models, self.num_gpus)).astype(np.float32)
        return WPRWorkflow(
            workflow_id,
            template,
            float(arrival),
            False,
            np.zeros(n, bool),
            np.zeros(n, bool),
            np.full(n, np.nan, np.float32),
            np.full(n, -1, np.int64),
            np.full(n, -1, np.int64),
            np.full(n, np.inf, np.float32),
            np.full(n, np.nan, np.float32),
            input_tokens,
            expected_out,
            actual_out,
            output_mb,
            exec_jitter,
        )

    def trace_stage_input_tokens(self, template: WPRTemplate, request_tokens: float, response_tokens: float) -> np.ndarray:
        """Map trace-level tokens to stage-level input tokens using template profiles."""

        vals = []
        for st in template.stages:
            name = st.name.lower()
            if st.execution_class != "llm":
                vals.append(max(64.0, 0.08 * request_tokens))
            elif "plan" in name:
                vals.append(max(64.0, 0.15 * request_tokens))
            elif "summary" in name or "salience" in name or "refine" in name:
                vals.append(max(128.0, 0.55 * request_tokens + 0.20 * response_tokens))
            elif "verify" in name or "check" in name:
                vals.append(max(64.0, 0.35 * response_tokens + 0.10 * request_tokens))
            elif "repair" in name:
                vals.append(max(128.0, 0.65 * request_tokens + 0.45 * response_tokens))
            else:
                vals.append(max(128.0, request_tokens + 0.25 * response_tokens))
        return np.asarray(vals, dtype=np.float32)

    def trace_stage_output_tokens(self, template: WPRTemplate, request_tokens: float, response_tokens: float) -> np.ndarray:
        """Map trace-level response tokens to stage-level output tokens."""

        vals = []
        for st in template.stages:
            name = st.name.lower()
            if st.execution_class != "llm":
                vals.append(16.0)
            elif "plan" in name:
                vals.append(max(16.0, 0.08 * response_tokens))
            elif "summary" in name or "salience" in name:
                vals.append(max(32.0, 0.25 * response_tokens))
            elif "verify" in name or "check" in name:
                vals.append(max(16.0, 0.05 * response_tokens))
            elif "repair" in name:
                vals.append(max(32.0, 0.45 * response_tokens))
            else:
                vals.append(max(32.0, response_tokens))
        return np.asarray(vals, dtype=np.float32)

    def select_trace_template(self, model_name: str, request_tokens: float, response_tokens: float) -> WPRTemplate:
        """Heuristic request-to-DAG mapping for trace-driven instantiation."""

        name = model_name.lower()
        by_name = {t.name: t for t in self.templates}
        if "code" in name or "repair" in name:
            return by_name["coding_repair" if response_tokens > 600 else "coding_success"]
        if request_tokens > 6000:
            return by_name["document_analysis"]
        if response_tokens > 1200 or "gpt-4" in name or "reason" in name:
            return by_name["deep_research"]
        return by_name["rag_qa"]

    def _generate_synthetic_arrivals(self) -> list[WPRWorkflow]:
        arrivals: list[WPRWorkflow] = []
        t = 0.0
        wid = 0
        probs = np.asarray([0.24, 0.25, 0.17, 0.14, 0.20])
        while t < self.horizon:
            t += float(self.rng.exponential(1.0 / max(self.arrival_rate, 1e-9)))
            if t >= self.horizon:
                break
            template = self.templates[int(self.rng.choice(len(self.templates), p=probs))]
            arrivals.append(self.instantiate_workflow(wid, template, t))
            wid += 1
        return arrivals

    def _earliest_unfinished_release_times(self, wf: WPRWorkflow) -> np.ndarray:
        pred_map = self._pred_map(wf.template)
        times = np.full(len(wf.template.stages), self.time, dtype=np.float32)
        for stage in wf.template.stages:
            i = stage.stage_id
            if wf.completed[i] or wf.scheduled[i]:
                times[i] = float(wf.finish_times[i])
            else:
                pred_ready = max([times[p] + self.communication_delay(wf, p, i) for p in pred_map[i]], default=max(self.time, float(wf.ready_times[i]) if np.isfinite(wf.ready_times[i]) else self.time))
                times[i] = pred_ready
                if stage.execution_class == "llm":
                    fastest = min(
                        (
                            m.prefill_time_per_ktok * stage.input_tokens_mean + m.decode_time_per_ktok * stage.output_tokens_mean + 0.18 * stage.work / max(g.speed for g in self.gpus)
                            for m in self.models
                            if self.model_feasible(stage, m)
                        ),
                        default=stage.work,
                    )
                    times[i] += 0.25 * fastest
                else:
                    times[i] += 0.25 * stage.tool_time_mean
        return times

    @staticmethod
    def _pred_map(template: WPRTemplate) -> dict[int, list[int]]:
        pred = {s.stage_id: [] for s in template.stages}
        for u, v in template.edges:
            pred[v].append(u)
        return pred

    @staticmethod
    def _succ_map(template: WPRTemplate) -> dict[int, list[int]]:
        succ = {s.stage_id: [] for s in template.stages}
        for u, v in template.edges:
            succ[u].append(v)
        return succ
