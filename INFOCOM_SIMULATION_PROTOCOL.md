# INFOCOM Simulation Protocol

This document defines the final trace-driven simulation protocol for WPRO. It is intended to prevent trace leakage and to make the experimental claims publication-safe.

## 1. Workload Setup

The simulator uses production-derived LLM request traces for:

- arrival timestamps;
- input token lengths;
- output token lengths;
- model/request type when available.

Each request is mapped to an agentic workflow DAG template:

```text
real LLM request trace + application workflow template
= trace-driven workflow instance
```

The trace does not natively contain complete agent workflow DAGs. The paper should state that the evaluation is a measurement-calibrated, trace-driven simulator under production-derived LLM workloads.

## 2. Chronological Data Split

Final trace-driven results must use chronological train/validation/test splits:

```text
60% training
20% validation
20% test
```

Requests are sorted by timestamp. The split must not randomly shuffle rows.

To prevent boundary leakage, a guard interval is removed around split boundaries:

```text
default guard interval = 30 seconds
```

The guard interval avoids workflows near one split boundary carrying temporal structure into the next split.

Use:

```powershell
py split_trace_dataset.py `
  --input data\public_traces\BurstGPT_1.csv `
  --output-dir data\public_traces\burstgpt_split `
  --timestamp-col Timestamp `
  --input-tokens-col "Request tokens" `
  --output-tokens-col "Response tokens" `
  --model-col Model `
  --drop-zero-output `
  --guard-seconds 30
```

The script outputs:

```text
trace_train.csv
trace_validation.csv
trace_test.csv
split_manifest.json
```

## 3. Training and Validation Protocol

WPR-A2C and all RL ablations are trained only on `trace_train.csv`.

Validation checkpoint selection uses only `trace_validation.csv`.

Final reported metrics are computed only on `trace_test.csv`.

The current runner supports this through:

```powershell
py run_wpr_trace_experiments.py `
  --train-trace-path data\public_traces\burstgpt_split\trace_train.csv `
  --validation-trace-path data\public_traces\burstgpt_split\trace_validation.csv `
  --test-trace-path data\public_traces\burstgpt_split\trace_test.csv `
  --timestamp-col Timestamp `
  --input-tokens-col "Request tokens" `
  --output-tokens-col "Response tokens" `
  --model-col Model `
  --checkpoint-metric weighted_completed_value `
  --validation-interval 5 `
  --validation-episodes 1 `
  --episodes 200 `
  --eval-episodes 20 `
  --seeds 5 `
  --output outputs\infocom_trace_burstgpt_final
```

## 4. Online Test Environment

During test:

- the trained policy parameters are fixed;
- no actor, critic, or demand-head update is allowed;
- arrivals follow the held-out test trace timestamps;
- token lengths are replayed from the held-out test trace;
- execution jitter is generated from fixed test seeds;
- all policies are evaluated on paired workload seeds.

## 5. Baselines

Final experiments should include:

- Random;
- EDF;
- Online ready greedy;
- DAG-oracle residency greedy;
- Vanilla A2C;
- WPR no progress;
- WPR no demand;
- WPR no residency;
- WPR no potential shaping;
- Full WPR-A2C.

For small-scale instances, include a strict optimal reference:

- CP-SAT or MILP if implemented;
- otherwise do not call the result strict optimality gap.

Bounded lookahead may be reported only as a bounded reference, not as optimal.

## 6. Metrics

Primary metrics:

- SLA success ratio;
- weighted completed value;
- weighted goodput rate;
- P95 latency.

Secondary metrics:

- completion ratio;
- admission ratio;
- rejected requests;
- dropped workflows;
- ready-stage waiting time;
- resident hit rate;
- model preparation time;
- demand prediction error.

## 7. Statistical Reporting

Final INFOCOM-grade reporting should use:

- 5 independent RL training seeds;
- at least 20 fixed test windows/configurations;
- paired workload evaluation across all policies;
- bootstrap 95% confidence intervals;
- Wilcoxon signed-rank test against the strongest online baseline.

The strongest online baseline should typically be the best of EDF, online greedy, and DAG-oracle greedy under the same test workload.

## 8. Figure Matrix

Final figures should be generated from CSV results, not manually edited. The paper-ready format is vector PDF/SVG; PNG is only for preview.

- grouped bar chart;
- workload-response line chart;
- latency-goodput scatter/Pareto chart;
- ablation dot plot;
- trace-driven bar chart;
- 2D convergence curve;
- 3D convergence surface.

Use:

```powershell
py generate_infocom_figures.py --input-dir outputs\infocom_trace_burstgpt_final
```

The script emits PDF, SVG, and PNG versions under `paper_figures/`. Figures use publication-style serif fonts, dashed grid lines, error bars, hatches/markers for grayscale readability, math-formatted axis labels, and external legends when needed.

## 9. Current Result Status

Existing repository results are preliminary.

They can be used for:

- checking code logic;
- designing INFOCOM figures;
- validating trend direction;
- selecting final experiment settings.

They should not yet be described as final publication results because earlier trace-driven runs did not use fully isolated train/validation/test trace files.

Observed preliminary trends:

- WPR-A2C is most useful under heavy load;
- light and moderate loads may be dominated by EDF or online greedy;
- current heavy synthetic workloads show stronger weighted-goodput behavior for WPR-A2C;
- dense BurstGPT-derived replay showed WPR-A2C goodput improvements, but this must be re-validated on held-out test trace splits.
