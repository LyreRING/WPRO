# Trace-Driven Evaluation Notes

## 为什么加入 trace-driven simulation

当前 WPRO simulator 已经能够完整模拟论文中的事件驱动 AIaaS workflow orchestration，包括 admission control、tool/LLM stage 区分、GPU model residency、cold loading、通信开销、token-dependent execution 和 SLA 统计。

真实部署全部 foundation models 并不是论文实验的必要条件。更合理、也更容易被审稿人接受的方式是：

```text
production-derived LLM request trace
+ measurement-calibrated model/GPU profile
+ agentic workflow DAG templates
= trace-driven AIaaS workflow orchestration simulation
```

这类实验应表述为 measurement-calibrated, trace-driven simulator under production-derived LLM workloads。

## 当前实现

`dag_a2c/wpr_env.py` 中新增：

- `WorkloadSource`：统一 workload 输入协议；
- `SyntheticWorkloadSource`：保留原 controlled synthetic workload；
- `TraceWorkloadSource`：从 CSV trace 中读取真实 arrival timestamp、input tokens、output tokens、model name 和可选 elapsed time；
- `WPREnv.instantiate_workflow()`：统一实例化 synthetic 或 trace-driven workflow；
- `WPREnv.select_trace_template()`：把 trace request 映射到 `rag_qa`、`deep_research`、`document_analysis`、`coding_*` 等 workflow DAG 模板；
- `trace_stage_input_tokens()` / `trace_stage_output_tokens()`：按 stage 语义把请求级 token 长度拆分到 workflow stages。

`run_wpr_trace_experiments.py` 是 trace-driven 实验入口，会输出：

- `episode_metrics.csv`
- `summary_metrics.csv`
- `training_curve.csv`
- `figures/weighted_completed_value.png`
- `figures/weighted_goodput_rate.png`
- `figures/sla_success_ratio.png`
- `figures/p95_latency.png`
- `figures/trace_characterization.png`

## CSV 字段

默认自动识别以下常见字段名：

- timestamp：`timestamp`, `time`, `arrival`, `created_at`
- input tokens：`request_tokens`, `input_tokens`, `prompt_tokens`
- output tokens：`response_tokens`, `output_tokens`, `completion_tokens`
- model：`model`, `model_name`, `engine`, `type`
- elapsed time：`elapsed_time`, `latency`, `duration`

如果真实 trace 字段名不同，可以通过命令行参数显式指定。

## 论文表述边界

推荐写法：

- The workload arrival process and token lengths are replayed from production-derived LLM traces.
- Each trace request is mapped to an agentic workflow template according to request/model characteristics.
- The auxiliary demand head estimates DAG-induced near-future model demand under uncertain stage durations.
- Synthetic workloads are used for controlled sensitivity experiments, while trace-driven workloads are used for realistic evaluation.

不建议写法：

- The trace contains complete real agent workflow DAGs.
- The demand predictor perfectly predicts actual future demand.
- The simulator is a full deployment of all foundation models.

## 后续接真实 trace 的建议

优先准备三类真实 trace：

1. 生产 LLM 请求到达 trace：timestamp、prompt tokens、completion tokens、model 或 request type。
2. 模型/GPU profiling：不同 model/GPU 上的 prefill/decode 时间、cold load 时间、显存占用。
3. 应用 workflow template：Deep Research、RAG QA、Coding Assistant、Document Analysis 等 DAG 模板和 stage token 比例。

实验图建议包括：

- Arrival rate sensitivity；
- Deadline tightness sensitivity；
- GPU heterogeneity；
- Cold load cost；
- Demand window；
- Demand predictor error；
- Resident hit rate；
- WAIT/admission behavior；
- Ablation: Progress, Demand, Residency, Potential Shaping, Vanilla A2C。

## 已接入的公开 trace

当前仓库已经支持 BurstGPT：

```powershell
py prepare_public_trace.py --download --mode dense --requests 120 --output data\public_traces\BurstGPT_1_dense_120.csv
```

本地已完成一轮公开 trace-driven pilot：

```text
outputs/wpr_trace_burstgpt_dense120_cap30_80ep
```

结果摘要见：

```text
EXPERIMENT_RESULTS_PUBLIC_TRACE.md
```
