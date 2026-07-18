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

## 已验证命令

```powershell
py -m py_compile dag_a2c\wpr_env.py dag_a2c\wpr_a2c.py dag_a2c\wpr_baselines.py run_wpr_experiments.py plot_wpr_results.py
py run_wpr_experiments.py --quick --output outputs\wpr_smoke_semantic_fix
py plot_wpr_results.py --input outputs\wpr_smoke_semantic_fix
```

## 注意

当前 smoke run 只用于工程验证，不能作为论文结论。论文级实验建议至少使用：

```powershell
py run_wpr_experiments.py --episodes 200 --eval-episodes 20 --seeds 5 --output outputs\wpr_main_200ep_5seed
```

如果后续需要严格 optimality gap，应单独实现小规模 MILP/CP-SAT；当前 `lookahead_gap` 只能解释为相对 bounded lookahead reference 的 gap。
