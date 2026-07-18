# WPR-A2C 实验说明

本目录中的 WPR-A2C 代码从 `dagA2C` 迁移到 `WPRO` 后，已按当前论文 system model 重构。

## 运行命令

快速工程检查：

```powershell
py run_wpr_experiments.py --quick --output outputs\wpr_smoke_semantic_fix
py plot_wpr_results.py --input outputs\wpr_smoke_semantic_fix
```

建议论文主实验：

```powershell
py run_wpr_experiments.py --episodes 200 --eval-episodes 20 --seeds 5 --output outputs\wpr_main_200ep_5seed
```

## 环境建模

`dag_a2c/wpr_env.py` 实现 event-driven AIaaS workflow orchestration：

- workflow arrival 时做规则化准入，输出 `rejected`；
- `execution_class="tool"` 的 stage 自动运行，不进入 GPU ready queue；
- `execution_class="llm"` 的 stage 由 orchestrator 选择 stage/model/GPU；
- GPU 使用 `IDLE_RESIDENT -> PREPARING -> RUNNING -> IDLE_RESIDENT` 状态机；
- successor ready time 包含 communication delay；
- LLM 执行时间由 input tokens、actual output tokens、model、GPU 和随机扰动决定；
- `ready_times` 真实参与 ready 判断和 `avg_ready_wait` 指标。

## WPR-A2C 模块

1. Workflow-progress encoder

   使用 permutation-invariant pooling 编码 active workflows 和 GPU residency，避免 active slot 移动导致状态语义不稳定。

2. Future Model-Demand Predictor

   使用当前 state 预测 `oracle_dag_demand_target(H)`。该标签来自 unfinished DAG 的窗口需求估计，不写作真实 rollout future。

3. Residency-aware cross scorer

   Actor logit 包含：

   ```text
   theta^T phi(S,a) + eta * DeltaPsi(S,a)
   ```

   其中 `phi(S,a)` 是每个候选动作独有的 workflow/stage/model/GPU/cross 特征。

4. Event-aware autoregressive matching decoder

   对 idle GPUs 按固定顺序选择动作，并支持 `WAIT_g=(-1,-1,-1,g)`。

5. Time-aware critic

   TD target 使用：

   ```text
   R_n + exp(-beta * Delta t_n) V(S_{n+1})
   ```

## Baselines

- `random`
- `edf`
- `online_greedy`：只看当前 ready queue；
- `dag_oracle_greedy`：可访问 DAG oracle demand 的强启发式；
- `vanilla_a2c`：真正训练的普通 A2C，不使用 progress/demand/residency 模块；
- WPR 消融：`wpr_no_progress`、`wpr_no_demand`、`wpr_no_residency`、`wpr_fixed_gamma`。

## 指标

- `weighted_completed_value`：完成且满足 SLA 的 workflow 权重和；
- `weighted_goodput_rate`：`weighted_completed_value / episode_time`；
- `sla_success_ratio`
- `completion_ratio`
- `p95_latency`
- `avg_ready_wait`
- `rejected`
- `dropped`
- `lookahead_gap`：相对 bounded lookahead reference，不是 strict optimality gap。
