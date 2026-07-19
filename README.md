# WPRO: WPR-A2C for Online AIaaS Workflow Orchestration

本仓库只保留当前课题思路的最终实验版本：Workflow-Progress and Residency-Aware A2C (WPR-A2C)。原始草稿、无关测试和旧版原型已经清理，当前代码围绕论文场景中的在线 AIaaS agentic workflow orchestration 展开。

## 文件组成

```text
WPRO/
├── dag_a2c/
│   ├── __init__.py
│   ├── wpr_env.py                 # 事件驱动 AIaaS workflow simulator
│   ├── wpr_a2c.py                 # WPR-A2C、Vanilla A2C 与消融配置
│   └── wpr_baselines.py           # Random、EDF、online greedy、DAG-oracle greedy、lookahead reference
├── data/
│   └── sample_trace_requests.csv  # trace-driven 入口的最小 smoke trace 示例
├── outputs/
│   ├── wpr_paper_v1_loads/        # 已跑出的 controlled synthetic v1 结果
│   └── wpr_trace_smoke/           # trace-driven smoke 结果
├── run_wpr_experiments.py         # controlled synthetic 实验入口
├── run_wpr_trace_experiments.py   # trace-driven 实验入口
├── plot_wpr_results.py            # 从 CSV 重新生成 controlled synthetic 图
├── docs_wpr_a2c_experiments.md    # 实验设计说明
├── TRACE_DRIVEN_EVALUATION.md     # trace-driven simulation 说明与论文表述边界
├── WPR_IMPLEMENTATION_STATUS.md   # 当前实现与论文 system model 的对应关系
├── EXPERIMENT_RESULTS_V1.md       # 已完成 v1 结果摘要
├── requirements.txt
└── WPRO.pptx
```

## 审阅顺序

1. `WPR_IMPLEMENTATION_STATUS.md`：看当前环境、动作空间、奖励、baseline 与论文模型的一致性。
2. `dag_a2c/wpr_env.py`：看 admission control、tool/LLM stage 区分、GPU residency 状态机、通信时延、token-dependent execution、预采样 service-time trace、trace workload loader。
3. `dag_a2c/wpr_a2c.py`：看 workflow-progress encoder、DAG-induced demand head、residency-aware action features、WAIT_ALL decoder、time-aware critic 和 GAE。
4. `run_wpr_experiments.py`：看 controlled synthetic 实验、消融、训练、评估、CSV 和图生成。
5. `run_wpr_trace_experiments.py`：看真实 trace timestamp/token 长度如何映射为 workflow 实例。
6. `EXPERIMENT_RESULTS_V1.md`：看当前 light/moderate/heavy 三档负载的第一版结果。

## 安装

```powershell
py -m pip install -r requirements.txt
```

## Controlled Synthetic 实验

快速验证：

```powershell
py run_wpr_experiments.py --quick --output outputs\wpr_smoke_final_fix
py plot_wpr_results.py --input outputs\wpr_smoke_final_fix
```

建议论文主实验：

```powershell
py run_wpr_experiments.py --episodes 200 --eval-episodes 20 --seeds 5 --output outputs\wpr_main_200ep_5seed
```

## Trace-Driven 实验

快速验证：

```powershell
py run_wpr_trace_experiments.py --trace-path data\sample_trace_requests.csv --quick --output outputs\wpr_trace_smoke
```

公开真实数据集 BurstGPT：

```powershell
py prepare_public_trace.py --download --mode dense --requests 120 --output data\public_traces\BurstGPT_1_dense_120.csv

py run_wpr_trace_experiments.py `
  --trace-path data\public_traces\BurstGPT_1_dense_120.csv `
  --timestamp-col Timestamp `
  --input-tokens-col "Request tokens" `
  --output-tokens-col "Response tokens" `
  --model-col Model `
  --deadline-mode relative `
  --deadline-multiplier 4.5 `
  --time-scale 1 `
  --max-requests 120 `
  --horizon 90 `
  --max-active 30 `
  --episodes 80 `
  --eval-episodes 5 `
  --seeds 2 `
  --output outputs\wpr_trace_burstgpt_dense120_cap30_80ep
```

替换为真实 trace 后的推荐形式：

```powershell
py run_wpr_trace_experiments.py `
  --trace-path data\real_llm_trace.csv `
  --timestamp-col timestamp `
  --input-tokens-col request_tokens `
  --output-tokens-col response_tokens `
  --model-col model `
  --elapsed-col elapsed_time `
  --deadline-mode relative `
  --deadline-multiplier 2.5 `
  --time-scale 10 `
  --max-requests 1000 `
  --episodes 200 `
  --eval-episodes 20 `
  --seeds 5 `
  --output outputs\wpr_trace_real_200ep_5seed
```

trace-driven 模式的含义是：

```text
真实 LLM 请求 trace 的到达时间和 token 长度
+ agentic application workflow DAG 模板
= trace-driven workflow instance
```

因此论文中建议称为 measurement-calibrated, trace-driven simulator under production-derived LLM workloads，而不是声称 trace 原生包含完整 agent workflow DAG。

## 指标说明

- `weighted_completed_value`：满足 SLA 的 workflow 权重和；
- `weighted_goodput_rate`：单位 episode 时间的加权完成收益；
- `sla_success_ratio`：到达请求中满足 SLA 的比例；
- `p95_latency`：完成 workflow 的 P95 latency；
- `avg_ready_wait`：LLM stage 从 ready 到开始执行的平均等待时间；
- `rejected`：arrival-time admission control 拒绝数量；
- `dropped`：运行中被判定不可行后丢弃的数量；
- `lookahead_gap`：相对 bounded lookahead reference 的 gap，不是 strict optimality gap。
