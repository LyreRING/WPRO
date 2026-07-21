# WPRO Paper Figure Data and Analysis Notes

当前 `paper_artifacts/figure_data` 下的 CSV 是论文图结构与展示风格的 draft data，用于审阅图形是否规范、指标是否一致、图例和数学符号是否匹配 WPRO 算法。最终投稿前应由严格 train / validation / test 隔离实验生成的 CSV 替换。

## Fig. 1 Overall Performance

Data: `paper_artifacts/figure_data/fig1_overall.csv`

Metrics: normalized utility \(U/U_{\mathrm{WPRO}}\)、admission ratio \(R_{\mathrm{adm}}\)、on-time ratio \(R_{\mathrm{on}}\)。

Expected takeaway: WPRO 不一定追求最高 admission ratio，而是在 admission 和 deadline-aware completion 之间取得更高 weighted utility。

## Fig. 2 Scalability

Data: `paper_artifacts/figure_data/fig2_scalability.csv`

Metric sweep: arrival rate \(\lambda\)。

Expected takeaway: 在低负载下各方法差距较小；随着负载升高，WPRO 通过 workflow-progress、future demand 和 residency-aware action scoring 保持更高 on-time ratio 和 utility。

## Fig. 3 Multi-environment Improvement

Data: `paper_artifacts/figure_data/fig3_robustness.csv`

Expected takeaway: 每个点表示一个 held-out workload setting，横轴是最强 baseline，纵轴是 WPRO。点落在 \(y=x\) 上方表示 WPRO 优于对应 setting 的最强 baseline。

## Fig. 4 Deadline and DAG Complexity Surface

Data: `paper_artifacts/figure_data/fig4_surface.csv`

Expected takeaway: 当 deadline 更紧、workflow DAG 更复杂时，单步 greedy 更容易被未来依赖和模型驻留耦合误导，WPRO 的相对收益更明显。

## Fig. 5 Latency Breakdown

Data: `paper_artifacts/figure_data/fig5_delay_breakdown.csv`

Components: queue waiting、model preparation、communication、execution。

Expected takeaway: WPRO 主要降低 queue waiting 和 model preparation，不应声称显著降低不可避免的 token execution time。

## Fig. 6 Demand Prediction and Residency

Data: `paper_artifacts/figure_data/fig6_prediction.csv`, `paper_artifacts/figure_data/fig6_residency.csv`

Expected takeaway: demand head 学习的是 \(d_m^{\mathrm{DAG}}(H)\) 的近未来估计；更准确的需求估计对应更高 resident hit ratio 和更少 full model load。

## Fig. 7 Ablation Study

Data: `paper_artifacts/figure_data/fig7_ablation.csv`

Variants: Full WPRO、w/o Progress、w/o Demand、w/o Residency、w/o Wait、w/o Shaping。

Expected takeaway: Demand 和 Residency 对 heavy-load utility 和 residency hit 影响最大；Wait 和 Shaping 主要影响 deadline pressure 下的 on-time completion 和训练稳定性。

## Fig. 8 Scheduling Behavior and Overhead

Data: `paper_artifacts/figure_data/fig8_timeline.csv`, `paper_artifacts/figure_data/fig8_overhead.csv`

Expected takeaway: Timeline 展示 WPRO 在事件驱动调度中保留 resident model、减少不必要 cold load；overhead 图展示 autoregressive matching decoder 的决策开销随 candidate set 近似线性增长，仍保持毫秒级。

## Final Replacement Rule

最终实验完成后，只替换 CSV，不改画图代码。所有图必须由以下命令重新生成：

```powershell
py generate_wpro_paper_figures.py --data-dir paper_artifacts\figure_data --output-dir paper_artifacts\figures
```
