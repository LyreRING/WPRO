# Baseline Adaptation and Unified Evaluation Framework

本文档整理 WPRO 论文实验部分的统一测试环境、baseline 适配、训练/验证/测试划分、指标定义和 8 张论文图的生成逻辑。

## 1. Unified Evaluation Environment

所有方法共享同一个 event-driven AIaaS simulator：

- 相同 workflow DAG 模板、stage type、deadline、service-class weight。
- 相同 model pool、GPU profile、memory capacity、cold-load cost、adapter/load state。
- 相同 tool / communication / LLM stage 区分。
- 相同 admission rule、arrival process、trace replay window。
- 相同 token-dependent deterministic expectation 和 sampled execution-time trace。

不同方法只替换 orchestration policy，包括 admission 后的 ready-stage selection、model selection、GPU assignment 和 WAIT decision。

## 2. Trace-driven Data Split

正式实验采用时间顺序划分，而不是随机打乱：

- Training: first 60% of the trace.
- Validation: next 20% of the trace.
- Test: last 20% of the trace.
- Guard interval: 默认在 split boundary 两侧删除 30 秒，避免一个 workflow 横跨两个数据集。

推荐命令：

```powershell
py split_trace_dataset.py --input data\public_trace.csv --output-dir data\trace_splits --train-ratio 0.6 --validation-ratio 0.2 --guard-interval 30
py run_wpr_trace_experiments.py --train-trace-path data\trace_splits\train.csv --validation-trace-path data\trace_splits\validation.csv --test-trace-path data\trace_splits\test.csv --episodes 120 --eval-episodes 20 --seeds 5 --output outputs\wpro_trace_final
py generate_infocom_figures.py --input-dir outputs\wpro_trace_final --output-dir outputs\wpro_trace_final\paper_figures
```

## 3. Baselines

| Baseline | Workflow Admission | Model Selection | GPU Assignment | WAIT | Notes |
| --- | --- | --- | --- | --- | --- |
| Random | Shared rule | Random feasible | Random idle GPU order | No | Sanity lower baseline |
| FCFS | Shared rule | Earliest local finish | Earliest admitted workflow first | No | Classical queueing baseline |
| EDF | Shared rule | Earliest local finish | Earliest workflow deadline first | No | Deadline-aware online baseline |
| SRPT | Shared rule | Earliest local finish | Least remaining critical-path work first | No | Shortest remaining processing-time style |
| Utility-Greedy | Shared rule | Current utility/slack/prep score | Greedy idle GPU matching | No | Current ready queue only |
| DAG-Oracle Greedy | Shared rule | Uses oracle DAG demand label | Greedy idle GPU matching | No | Strong oracle heuristic, not normal online |
| Vanilla A2C | Shared rule | Learned | Learned | No | Same A2C/GAE, no WPRO modules |
| PPO | Shared rule | Learned | Learned | Configurable | To be added for final RL baseline |
| Lyapunov | Shared rule | Drift-plus-penalty score | Greedy matching | Optional | To be added for final queueing-control baseline |
| FCFS-vLLM / EDF-vLLM / SRPT-vLLM | Shared rule | Continuous batching inspired | GPU queue rule | No | Adapted baseline, not full vLLM system |
| FCFS-Orca | Shared rule | Iteration-level serving inspired | GPU queue rule | No | Adapted baseline, not full Orca runtime |
| FCFS-Sarathi | Shared rule | Chunked-prefill inspired | GPU queue rule | No | Adapted baseline, not full Sarathi runtime |
| WPRO | Shared rule | Workflow-progress and residency-aware actor | Event-aware autoregressive decoder | Yes | Main proposed method |

论文中必须强调：除非完整接入真实系统，否则 vLLM / Orca / Sarathi 只能称为 inspired/adapted baselines。

## 4. Metrics

- Admission ratio: \(R_{\mathrm{adm}} = N_{\mathrm{admitted}} / N_{\mathrm{arrived}}\)。该值不是越高越好，需要和 on-time ratio 一起看。
- On-time completion ratio: \(R_{\mathrm{on}} = N_{\mathrm{completed~before~deadline}} / N_{\mathrm{arrived}}\)。
- Conditional SLA success: \(R_{\mathrm{sla|adm}} = N_{\mathrm{completed~before~deadline}} / N_{\mathrm{admitted}}\)。
- Weighted completed value: \(V_{\mathrm{w}} = \sum_j w_j I_j^{\mathrm{on-time}}\)。
- Weighted goodput rate: \(G_{\mathrm{w}} = V_{\mathrm{w}} / T_{\mathrm{episode}}\)。
- P95 latency: 95th percentile of admitted workflow completion latency.
- Residency hit ratio: fraction of dispatched LLM stages whose model is already resident.
- Loading overhead: number and time of full model preparations.
- Queue waiting: ready-to-start latency accumulated after a stage becomes executable.
- Communication overhead: placement-dependent transfer delay before successor readiness.
- Predictor error: MAE/MSE between \(\hat d_m(H)\) and \(d_m^{\mathrm{DAG}}(H)\).
- Decision overhead: policy inference time per orchestration event.

## 5. Statistical Protocol

正式 INFOCOM-style result 建议：

- 5 independent RL training seeds.
- 每个配置至少 20 fixed held-out test windows。
- Validation checkpoint selection，不在 test 上调参。
- Paired workload evaluation：同一个 test window 同时跑所有 baselines。
- Bootstrap 95% confidence interval。
- 与 strongest online baseline 做 Wilcoxon signed-rank test。
- 小规模 optimality gap 只在 CP-SAT/MILP 或完整枚举后报告。

## 6. Eight Figure Plan

| Figure | File stem | Data CSV | Purpose |
| --- | --- | --- | --- |
| Fig. 1 | `fig1_overall_performance` | `fig1_overall.csv` | Overall utility, admission, on-time performance |
| Fig. 2 | `fig2_scalability_vs_arrival_rate` | `fig2_scalability.csv` | Robustness under increasing arrival rate |
| Fig. 3 | `fig3_multi_environment_improvement` | `fig3_robustness.csv` | Improvement over strongest baseline across workloads |
| Fig. 4 | `fig4_deadline_complexity_surface` | `fig4_surface.csv` | Deadline tightness and DAG complexity interaction |
| Fig. 5 | `fig5_latency_breakdown` | `fig5_delay_breakdown.csv` | Queue, preparation, communication, execution decomposition |
| Fig. 6 | `fig6_demand_prediction_and_residency` | `fig6_prediction.csv`, `fig6_residency.csv` | Demand estimator quality and residency mechanism |
| Fig. 7 | `fig7_ablation_study` | `fig7_ablation.csv` | Contribution of Progress, Demand, Residency, WAIT, Shaping |
| Fig. 8 | `fig8_schedule_timeline_and_overhead` | `fig8_timeline.csv`, `fig8_overhead.csv` | Scheduling behavior and decision overhead |

Draft figures can be generated by:

```powershell
py generate_wpro_paper_figures.py --create-draft-data
```

The draft CSV values are for figure-design review only. Before submission, replace them with final CSVs produced from held-out trace experiments.
