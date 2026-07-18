# WPRO: WPR-A2C for Online AIaaS Workflow Orchestration

本仓库只保留当前课题思路的最终实验版本：Workflow-Progress and Residency-Aware A2C (WPR-A2C)。

## 文件组成

```text
WPRO/
├── dag_a2c/
│   ├── __init__.py
│   ├── wpr_env.py          # 事件驱动 AIaaS workflow simulator
│   ├── wpr_a2c.py          # WPR-A2C、vanilla A2C 与消融配置
│   └── wpr_baselines.py    # EDF、在线贪心、DAG-oracle 贪心、lookahead reference
├── outputs/
│   ├── wpr_smoke_final_fix/
│   └── wpr_paper_v1_loads/
│       ├── episode_metrics.csv
│       ├── summary_metrics.csv
│       ├── training_curve.csv
│       └── figures/        # 由 CSV 真实生成的 smoke 图
├── run_wpr_experiments.py  # 训练、评估、汇总、绘图一体入口
├── plot_wpr_results.py     # 只读取 CSV 重新生成图
├── docs_wpr_a2c_experiments.md
├── WPR_IMPLEMENTATION_STATUS.md
├── EXPERIMENT_RESULTS_V1.md
├── requirements.txt
└── WPRO.pptx
```

## 推荐审阅顺序

1. `WPR_IMPLEMENTATION_STATUS.md`

   查看当前实现与论文 system model 的对应关系，以及哪些指标不能被误写成 strict optimal。

2. `dag_a2c/wpr_env.py`

   审阅准入控制、工具阶段、GPU residency 状态机、通信时延、token-dependent execution、预采样公平 service-time trace 与 ready-time 维护。

3. `dag_a2c/wpr_a2c.py`

   审阅 WPR-A2C 的五个模块：workflow-progress encoder、DAG-induced demand head、residency-aware action features、WAIT_ALL autoregressive matching decoder、time-aware MLP critic 与 event-aware GAE。

4. `dag_a2c/wpr_baselines.py`

   审阅 baseline 定义，尤其区分 `online_greedy` 与 `dag_oracle_greedy`。

5. `run_wpr_experiments.py`

   审阅实验入口、随机种子、quick 模式、指标汇总和图生成逻辑。

6. `EXPERIMENT_RESULTS_V1.md`

   查看当前 light/moderate/heavy 三档负载的第一版论文实验结论。

## 运行方式

安装依赖：

```powershell
py -m pip install -r requirements.txt
```

快速工程验证：

```powershell
py run_wpr_experiments.py --quick --output outputs\wpr_smoke_final_fix
py plot_wpr_results.py --input outputs\wpr_smoke_final_fix
```

建议论文主实验：

```powershell
py run_wpr_experiments.py --episodes 200 --eval-episodes 20 --seeds 5 --output outputs\wpr_main_200ep_5seed
```

## 指标说明

- `weighted_completed_value`：满足 SLA 的 workflow 权重和；
- `weighted_goodput_rate`：单位 episode 时间的加权完成收益；
- `sla_success_ratio`：到达请求中满足 SLA 的比例；
- `p95_latency`：完成 workflow 的 P95 latency；
- `avg_ready_wait`：LLM stage 从 ready 到开始执行的平均等待时间；
- `rejected`：arrival-time admission control 拒绝数量；
- `dropped`：运行中被判定不可行后丢弃数量；
- `lookahead_gap`：相对 bounded lookahead reference 的 gap，不是 strict optimality gap。
