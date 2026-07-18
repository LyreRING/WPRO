# WPR-A2C 实验实现状态

当前工作目录：`C:\Users\11634\Documents\Learn\Paper\WPRO`

## 已对齐论文逻辑的部分

1. 准入控制

   环境不再把请求放进无限隐式等待缓冲区。`WPREnv._process_arrivals()` 在 workflow arrival time 立刻做 admission decision：

   ```text
   x_j = 1{ |J_act| < N_max 且 deterministic SLA feasibility 成立 }
   ```

   被拒绝的 workflow 进入 `rejected_workflows`，实验指标输出 `rejected`。

2. LLM stage 与外部工具阶段分离

   `WPRStage.execution_class` 支持：

   - `llm`：进入 GPU ready queue，由 orchestrator 选择 stage/model/GPU；
   - `tool`：前驱完成并满足 input availability 后自动启动，不参与 GPU/model 选择；
   - `communication`：当前通过边上的输入传输时延体现，不单独调度。

3. GPU residency 状态机

   每台 GPU 维护：

   - `resident_model[g]`：已经可执行的 resident model；
   - `target_model[g]`：正在准备或运行绑定的 model；
   - `model_ready_time[g]`：cold load / adapter load 完成时间；
   - `gpu_state[g]`：`IDLE_RESIDENT / PREPARING / RUNNING`。

4. Stage communication cost

   后继 stage 的 `ready_time` 会加入前驱输出传输时间：

   ```text
   L_com = network_latency + output_size / bandwidth
   ```

   若前驱是工具阶段或跨 server，则使用不同估计。

5. Token-dependent execution

   每个 stage instance 生成：

   - `input_tokens`
   - `expected_output_tokens`
   - `actual_output_tokens`
   - `output_mb`

   LLM 执行时间采用：

   ```text
   L_exec = alpha_{m,g} * N_in + beta_{m,g} * N_out + semantic_work + jitter
   ```

   实现上已拆分为：

   - `expected_exec_time()`：确定性期望执行时间，用于 Actor action features、baselines、lookahead 和准入估计；
   - `sample_exec_time()`：只在真实 `_schedule_llm()` 时使用。

   为保证 baseline 公平性，workflow 生成时会预采样每个 `(stage, model, gpu)` 的 `exec_jitter` counterfactual table。不同算法在相同 seed 下共享同一组潜在服务时间，不会因为候选动作遍历次数不同而消耗不同随机数。

6. ready_times 真实使用

   `ready_times` 由 arrival/source stage 和通信完成时间更新，用于：

   - ready-stage feasibility；
   - `avg_ready_wait` 指标；
   - workflow-progress 特征中的 ready wait。

7. Coding workflow 语义

   coding workflow 被展开为两类 DAG：

   - `coding_success`
   - `coding_repair`

   避免把 repair 写成必然执行且依赖关系不合理的固定 DAG。

8. Baselines 修正

   - `vanilla_a2c`：真正训练的普通 A2C 配置，不再是确定性启发式；
   - `online_greedy`：只看当前 ready queue 的真实在线贪心；
   - `dag_oracle_greedy`：显式标注为可访问 DAG oracle demand 的强启发式；
   - `lookahead_search_upper_reference`：显式命名为 bounded lookahead reference，不称 optimal。

9. 工程入口修正

   - `--quick --output <path>` 会保留用户指定输出目录；
   - policy seed 使用固定 `POLICY_SEED_OFFSET`，不再使用 Python `hash(name)`；
   - 默认训练轮数提高到 80 episodes；
   - 输出新增 `weighted_completed_value` 和 `weighted_goodput_rate`。

10. 奖励与 credit assignment

   环境加入 potential-based reward shaping：

   ```text
   r_shape = r_terminal + exp(-beta * Delta t) Phi(S') - Phi(S)
   ```

   `Phi(S)` 根据 active workflow 的完成进度、剩余关键路径和 deadline slack ratio 构造：

   ```text
   Phi(S)=sum_j w_j * alpha_j * clip(slack_j / D_j, 0, 1)
   ```

   因此完成 stage 会提高 potential，单纯等待导致 slack 下降时不会被反向奖励，逾期 workflow 的 potential 为 0。最终优化目标仍是 SLA-compliant weighted value，但中间 stage completion 和 slack 变化现在会产生学习信号。`enable_potential_shaping` 可关闭该项，用于 `wpr_no_shaping` 消融。

11. WAIT 与事件边界

   per-GPU WAIT 已替换为全局 `WAIT_ALL=(-1,-1,-1,-1)`。如果选择 dispatch，decoder 会继续为当前 idle GPUs 构造 assignment set；只有决定本轮完全不调度时才推进到下一个外生事件。`WAIT_ALL` 使用全局 GPU 聚合特征，不再绑定第一张 idle GPU。

   环境增加了零时间 no-op 保护：若 `dt=0` 且离散状态签名完全不变，将直接报错，避免依赖 step limit 掩盖事件边界错误。

12. Actor/Critic 更新

   固定 residency scorer 不再作为额外 logit 加分项，而是进入 action feature 由 Actor 学习。Critic 从线性函数升级为两层 MLP。训练从单步 TD 改为 event-aware GAE，并采用“先冻结 value 计算整条 rollout 的 advantage/return，再统一更新 Actor/Critic”的标准流程。

## 已验证命令

```powershell
py -m py_compile dag_a2c\wpr_env.py dag_a2c\wpr_a2c.py dag_a2c\wpr_baselines.py run_wpr_experiments.py plot_wpr_results.py
py run_wpr_experiments.py --quick --output outputs\wpr_smoke_final_fix
py plot_wpr_results.py --input outputs\wpr_smoke_final_fix
```

## 注意

当前 smoke run 只用于工程验证，不能作为论文结论。论文级实验建议至少使用：

```powershell
py run_wpr_experiments.py --episodes 200 --eval-episodes 20 --seeds 5 --output outputs\wpr_main_200ep_5seed
```

如果后续需要严格 optimality gap，应单独实现小规模 MILP/CP-SAT；当前 `lookahead_gap` 只能解释为相对 bounded lookahead reference 的 gap。
