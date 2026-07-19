# Public Trace-Driven Experiment Results

## Dataset

本轮使用公开真实 LLM serving trace：BurstGPT `BurstGPT_1.csv`。

BurstGPT 是 Azure 支撑的真实 ChatGPT/GPT-4 serving workload trace。仓库版 CSV 字段包括：

- `Timestamp`
- `Model`
- `Request tokens`
- `Response tokens`
- `Total tokens`
- `Log Type`

由于仓库版 `data/BurstGPT_1.csv` 不含新版 release 中的 `Elapsed time` 和 `Session ID` 字段，本轮实验使用真实 arrival timestamp、model name、request tokens 和 response tokens；deadline 按 workflow template 的 relative fastest duration 生成。

## Trace Preparation

原始 trace 行数约 1.43M。为得到适合在线调度压力测试、但又不被 admission rejection 完全支配的负载片段，本轮从过滤掉 `Response tokens = 0` 的请求中选择一个真实连续 120-request / 30-second burst window：

```powershell
py prepare_public_trace.py --download --mode span --requests 120 --target-span 30 --output data\public_traces\BurstGPT_1_span30s_120.csv
```

实际选择窗口：

```text
start timestamp = 822291
end timestamp   = 822321
span            = 30 seconds
requests        = 120
```

这代表真实生产 trace 中的 burst window，而不是重新采样的 Poisson arrival。

## Main Pilot Command

```powershell
py run_wpr_trace_experiments.py `
  --trace-path data\public_traces\BurstGPT_1_span30s_120.csv `
  --timestamp-col Timestamp `
  --input-tokens-col "Request tokens" `
  --output-tokens-col "Response tokens" `
  --model-col Model `
  --deadline-mode relative `
  --deadline-multiplier 4.5 `
  --time-scale 1 `
  --max-requests 120 `
  --horizon 140 `
  --max-active 30 `
  --episodes 30 `
  --eval-episodes 3 `
  --seeds 1 `
  --checkpoint-metric weighted_completed_value `
  --validation-interval 5 `
  --validation-episodes 1 `
  --output outputs\wpr_trace_burstgpt_span30_validation_probe
```

## Key Results

| Policy | Weighted value | Goodput rate | SLA ratio |
|---|---:|---:|---:|
| Random | 32.00 | 0.767 | 0.0833 |
| EDF | 54.40 | 1.312 | 0.1417 |
| Online greedy | 65.07 | 1.625 | 0.1694 |
| DAG-oracle greedy | 52.27 | 1.264 | 0.1361 |
| Vanilla A2C | 32.00 | 0.767 | 0.0833 |
| WPR no progress | 74.67 | 1.866 | 0.1944 |
| WPR no demand | 54.40 | 1.319 | 0.1417 |
| WPR no residency | 55.47 | 1.336 | 0.1444 |
| WPR no shaping | 54.40 | 1.334 | 0.1417 |
| WPR-A2C | 96.00 | 2.395 | 0.2500 |

## Interpretation

这轮公开 trace-driven pilot 符合论文预期：

- WPR-A2C 的 `weighted_completed_value`、`weighted_goodput_rate` 和 `SLA success ratio` 三项均最高；
- WPR-A2C 高于 online greedy，说明在 production-derived burst workload 下，结构化长期调度比纯 immediate greedy 更有价值；
- WPR-A2C 显著优于 Vanilla A2C，说明改进不是单纯来自“用了 RL”；
- 去掉 demand、residency 或 potential shaping 后都有明显下降，支持模块消融论点；
- `wpr_no_progress` 仍然强于传统 baselines，说明 action-specific stage/model/GPU 结构先验本身有效；full WPR-A2C 进一步结合 progress、demand 和 residency 获得最高收益。

论文中建议把这组结果作为 `production-derived burst workload` 主图之一，并与 controlled synthetic 的 arrival-rate、deadline tightness、GPU heterogeneity、cold-load sensitivity 图配合使用。
