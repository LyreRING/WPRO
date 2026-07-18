# WPR-A2C 实验说明

本目录中的 WPR-A2C 代码从 `dagA2C` 迁移到 `WPRO` 后，已按当前论文 system model 重构。

## 运行命令

快速工程检查：

```powershell
py run_wpr_experiments.py --quick --output outputs\wpr_smoke_final_fix
py plot_wpr_results.py --input outputs\wpr_smoke_final_fix
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
- LLM 期望执行时间由 input tokens、actual output tokens、model 和 GPU 决定；真实执行时长从预采样 `exec_jitter(stage, model, gpu)` counterfactual table 中读取，保证不同算法在相同 seed 下共享公平随机 trace；
- `ready_times` 真实参与 ready 判断和 `avg_ready_wait` 指标。

## WPR-A2C 模块

1. Workflow-progress encoder

   使用 permutation-invariant pooling 编码 active workflows 和 GPU residency，避免 active slot 移动导致状态语义不稳定。

2. DAG-induced Model-Demand Head

   使用当前 state 学习 `oracle_dag_demand_target(H)`。当前实现更严谨地表述为 DAG-induced near-future model-demand estimation under uncertain stage durations，而不是 actual unknown future 的 perfect prediction。

3. Residency-aware cross scorer

   Actor logit 不再额外硬加固定 residency 分数，而是把 residency delta、prep time、resident hit、target/current demand 等作为 action feature：

   ```text
   z_theta(S,a) = theta^T phi(S,a)
   ```

   其中 `phi(S,a)` 是每个候选动作独有的 workflow/stage/model/GPU/cross 特征。

4. Event-aware autoregressive matching decoder

   先判断是否执行全局 `WAIT_ALL=(-1,-1,-1,-1)`。若不等待，则按固定顺序为 idle GPUs 构造 assignment set，并尽量处理完当前可行匹配，避免同一 timestamp 反复重新决策。

5. Time-aware MLP critic and GAE

   Critic 使用两层 MLP。训练时使用 event-aware GAE：

   ```text
   delta_n = r_n + exp(-beta * Delta t_n) V(S_{n+1}) - V(S_n)
   A_n = delta_n + exp(-beta * Delta t_n) lambda A_{n+1}
   ```

   实现中先用冻结的 critic 计算整条 rollout 的 `V_n`、`A_n` 和 return，再统一更新 Actor/Critic。

## Baselines

- `random`
- `edf`
- `online_greedy`：只看当前 ready queue；
- `dag_oracle_greedy`：可访问 DAG oracle demand 的强启发式；
- `vanilla_a2c`：真正训练的普通 A2C，使用相同 Actor/Critic/GAE/训练轮数/环境，但关闭 workflow-progress encoder、demand head、residency action features 和 WAIT action；
- WPR 消融：`wpr_no_progress`、`wpr_no_demand`、`wpr_no_residency`、`wpr_fixed_gamma`、`wpr_no_shaping`。

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
## Trace-driven realistic evaluation

除 controlled synthetic workload 之外，当前代码已经提供 trace-driven evaluation 入口：

```powershell
py run_wpr_trace_experiments.py --trace-path data\sample_trace_requests.csv --quick --output outputs\wpr_trace_smoke
```

真实 trace 建议至少包含：

- `timestamp`：真实请求到达时间；
- `request_tokens` / `input_tokens`：输入 token 长度；
- `response_tokens` / `output_tokens`：输出 token 长度；
- `model`：请求使用或偏好的模型名称，可选；
- `elapsed_time`：原始请求端到端延迟，可选，用于 deadline calibration。

实验方法：

```text
real request trace + application workflow template = trace-driven workflow instance
```

也就是说，真实 trace 提供生产负载的到达过程和 token 分布，workflow DAG 由应用模板给出。该设置适合作为论文的 realistic evaluation；controlled synthetic workload 继续用于研究 arrival rate、deadline tightness、GPU heterogeneity、cold load cost 等因素的可控敏感性。
