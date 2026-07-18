# WPR-A2C Experiment Results v1

本文件记录当前已完成的第一版论文实验结果。当前结论来自：

```text
outputs/wpr_paper_v1_loads/
```

该目录合并了：

- light：`outputs/wpr_main_v2_light_240ep_2seed`
- moderate：`outputs/wpr_main_v2_moderate_240ep_2seed`
- heavy：`outputs/wpr_main_v2_mh_120ep_2seed`

## 当前结论

1. Light workload

   WPR-A2C 与 EDF / online greedy 基本持平。该场景负载较轻，长期 residency/value planning 的收益空间有限，简单启发式已经接近饱和。

2. Moderate workload

   WPR-A2C 在 `weighted_completed_value` 上超过 EDF、online greedy 和 vanilla A2C，仅略低于 DAG-oracle greedy；在 `sla_success_ratio` 上排名第一。说明中等竞争强度下，workflow-progress、demand 和 residency-aware 表示开始体现价值。

3. Heavy workload

   WPR-A2C 在 `weighted_completed_value`、`weighted_goodput_rate` 和 `sla_success_ratio` 上均排名第一，相比 EDF、online greedy、DAG-oracle greedy 和 vanilla A2C 有明显优势。该场景最能支撑论文主论点：高负载下 workflow evolution 与 model residency 的长期耦合不能被短视启发式充分利用。

## 重要观察

- `wpr_with_wait` 当前不如稳健版 `wpr_a2c`。这说明在当前 workload 生成器中，显式等待动作不是主要收益来源，甚至可能带来机会成本。论文中建议把 WAIT 作为行为分析/消融项，而不要夸大其贡献。
- `wpr_no_demand` 在 heavy 场景表现较强，但仍低于完整 WPR-A2C。该结果支持 demand representation 的贡献，但后续需要更多 seeds 强化统计显著性。
- `vanilla_a2c` 是真正训练的 A2C baseline，使用相同 Actor/Critic/GAE/环境，但关闭 workflow-progress、demand、residency 和 WAIT 特征。

## 已生成图

```text
outputs/wpr_paper_v1_loads/figures/weighted_completed_value.png
outputs/wpr_paper_v1_loads/figures/weighted_goodput_rate.png
outputs/wpr_paper_v1_loads/figures/sla_success_ratio.png
outputs/wpr_paper_v1_loads/figures/p95_latency.png
outputs/wpr_paper_v1_loads/figures/wpr_training_diagnostics.png
```

## 后续正式实验建议

当前 v1 结果已经能支撑主趋势，但正式论文仍建议继续补：

- 5 到 10 个 random seeds；
- deadline tightness sweep；
- arrival rate sweep；
- cold-load / preparation-cost sensitivity；
- GPU heterogeneity sensitivity；
- demand window sensitivity；
- demand head prediction error；
- resident hit rate；
- preparation overhead distribution；
- admission ratio 和 ready waiting 分析。

这些实验应作为论文最终图表扩展，而不是继续改变核心算法。
