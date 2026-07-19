# Public Trace-Driven Experiment Results

## Dataset

本轮使用公开真实 LLM serving trace：BurstGPT `BurstGPT_1.csv`。

BurstGPT 是 Azure 支撑的真实 ChatGPT/GPT-4 serving workload trace，字段包括：

- `Timestamp`
- `Model`
- `Request tokens`
- `Response tokens`
- `Total tokens`
- `Log Type`

由于仓库版 `data/BurstGPT_1.csv` 不含新版 release 中的 `Elapsed time` 和 `Session ID` 字段，本轮实验使用真实 arrival timestamp、model name、request tokens 和 response tokens，deadline 仍按 workflow template 的 relative fastest duration 生成。

## Trace Preparation

原始 trace 行数约 1.43M。为得到适合在线调度压力测试的高负载片段，本轮从过滤掉 `Response tokens = 0` 的请求中选择最密集的 120-request window：

```powershell
py prepare_public_trace.py --download --mode dense --requests 120 --output data\public_traces\BurstGPT_1_dense_120.csv
```

实际选择窗口：

```text
start timestamp = 3064252
end timestamp   = 3064256
span            = 4 seconds
requests        = 120
```

这代表真实生产 trace 中的 burst window，而不是重新采样的 Poisson arrival。

## Main Pilot Command

```powershell
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

## Key Results

| Policy | Weighted value | Goodput rate | SLA ratio |
|---|---:|---:|---:|
| Random | 21.00 | 1.065 | 0.0417 |
| EDF | 35.70 | 1.969 | 0.0708 |
| Online greedy | 37.38 | 2.032 | 0.0742 |
| DAG-oracle greedy | 39.48 | 2.092 | 0.0783 |
| Vanilla A2C | 15.96 | 0.789 | 0.0317 |
| WPR no progress | 29.40 | 1.527 | 0.0583 |
| WPR no demand | 25.20 | 1.222 | 0.0500 |
| WPR no residency | 20.16 | 1.053 | 0.0400 |
| WPR no shaping | 5.46 | 0.241 | 0.0108 |
| WPR-A2C | 36.54 | 2.136 | 0.0725 |

## Interpretation

这轮公开 trace-driven pilot 的结论比较稳：

- WPR-A2C 的 `weighted_goodput_rate` 最高，高于 DAG-oracle greedy、online greedy 和 EDF；
- WPR-A2C 的 weighted value 接近 greedy/oracle reference，但不是最高；
- WPR-A2C 显著优于 Vanilla A2C，说明改进不是单纯来自“用了 RL”；
- 去掉 progress、demand、residency 或 potential shaping 后都有明显下降，支持模块消融论点；
- 该 burst window 极端拥塞，admission rejection 很高，因此它更适合作为 high-load burst stress test，而不是唯一主实验。

论文中建议把这组结果作为 `production-derived burst workload`，并与 controlled synthetic 的 arrival-rate/deadline/GPU heterogeneity sensitivity 图配合使用。
