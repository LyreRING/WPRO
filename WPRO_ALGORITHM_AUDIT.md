# WPRO Algorithm Audit

本文档用于把论文中的 WPRO 算法模块和当前工程代码对应起来，避免后续写论文或答 reviewer 时出现“描述强于实现”的问题。

## 1. 算法名称

论文中统一使用 **WPRO**，全称为 **Workflow-Progress and Residency-aware Orchestration**。

代码中历史类名仍保留 `WPRA2CConfig` 和 `train_wpr_agent`，它们对应 WPRO 的 A2C 实现版本。论文图例、实验表格和正文不再使用 `WPR-A2C` 作为主算法名。

## 2. 训练与推理方式

WPRO 不是每次测试从零开始在线训练，而是：

1. 在训练 trace 或 synthetic workload 上离线训练 actor、critic 和 demand auxiliary head。
2. 在 validation split 上选择 checkpoint。
3. 在 held-out test split 上固定参数推理。
4. 测试时仍是 online decision：调度器只能看到当前系统状态和 DAG 信息，不能访问 test trace 的未来到达。

因此，正式论文结果必须使用 train / validation / test 隔离，不能在同一条 trace 上同时训练和测试。

## 3. 模块对应关系

| 论文模块 | 当前代码位置 | 当前状态 |
| --- | --- | --- |
| Workflow-progress encoder | `dag_a2c/wpr_a2c.py` 中状态编码和 workflow progress features | 已实现 |
| Future model-demand estimator | `FutureModelDemandPredictor` 与 `oracle_dag_demand_target()` 辅助标签 | 已实现，但应表述为 DAG-induced near-future demand estimation |
| Residency-aware action representation | `action_features()`、residency hit、prep time、resident replacement demand features | 已实现 |
| Action-specific actor | softmax over candidate-specific feature vector `phi(S,a)` | 已实现 |
| Event-aware matching decoder | `dispatch()` 按 idle GPU 自回归选择动作，并支持 WAIT | 已实现 |
| Time-aware critic and GAE | `TimeAwareCritic`、`update_from_gae()`、`exp(-beta * dt)` | 已实现 |
| Potential shaping | 环境 potential function 和 `use_potential_shaping` 消融开关 | 已修正 |
| Validation checkpoint | `train_wpr_agent(..., validation_env_factory=...)` | 已实现 |

## 4. 必须谨慎表述的点

- Demand predictor 不是“完美预测真实未来请求”，而是估计由当前未完成 DAG、估计执行时间和 deadline slack 诱导出的 near-future model demand。
- DAG-Oracle Greedy 是强启发式上界基线，它访问 oracle DAG demand label，不能作为普通在线算法。
- Lookahead reference 是 bounded lookahead / beam-search reference，不是严格 MILP optimal。若论文报告 optimality gap，需要另外实现 CP-SAT/MILP 或完整小规模枚举。
- vLLM、Orca、Sarathi 不应写成完整系统复现实验，除非真的接入对应 runtime。更合适的写法是 unified simulator 下的 inspired/adapted scheduling baselines。

## 5. 当前优先级

核心环境、动作空间、actor、critic、WAIT、potential shaping、trace split 和 validation checkpoint 已经闭合。接下来优先事项是：

1. 用隔离 trace split 进行 5 个 RL training seeds 和至少 20 个 paired test windows。
2. 补齐 PPO、Lyapunov、FCFS-vLLM / EDF-vLLM / SRPT-vLLM 等适配 baseline。
3. 对最终 CSV 运行 `generate_wpro_paper_figures.py`，替换 draft figure data。
