# Usage Reference

This document is the complete operator-facing usage reference for Seldon.

It explains:

- how to build,
- how to run,
- how to configure,
- how to tune,
- and how to troubleshoot.

No prior repository context is required.

---

## 1) Build and Run Basics

### 1.1 Build

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

Float32 neural tensors (reduced memory):

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON -DSELDON_NEURAL_FLOAT32=ON ..
cmake --build . -j
```

### 1.2 Direct Run

```bash
./seldon /absolute/path/to/data.csv
```

### 1.3 Run With Config

```bash
./seldon /absolute/path/to/data.csv --config /absolute/path/to/seldon.yaml
```

### 1.4 Interactive Setup

```bash
./seldon --interactive
```

### 1.5 Massif Memory Profiling

```bash
valgrind --tool=massif --massif-out-file=massif.out \
  ./seldon /absolute/path/to/data.csv --low-memory true --neural-streaming-mode true

ms_print massif.out > massif_report.txt
```

If `valgrind` is unavailable, install it first (distribution package name is usually `valgrind`).

---

## 2) Input Sources and Prerequisites

Supported input formats:

- `.csv`
- `.csv.gz`
- `.csv.zip`
- `.xlsx`
- `.xls`

Tooling used for non-plain CSV inputs:

- `gzip` for `.csv.gz`
- `unzip` for `.csv.zip`
- `xlsx2csv` for `.xlsx`
- `xls2csv` for `.xls`

Optional but useful tools:

- `gnuplot` for plots
- `pandoc` for HTML conversion

---

## 3) Command Structure

General pattern:

```bash
./seldon <dataset_path> [options]
```

Primary positional input:

- `dataset_path`

Option style:

- `--key value`

Examples:

```bash
./seldon /data/earthquakes.csv --target magnitude
./seldon /data/phone_addiction.xlsx --plots none
./seldon /data/data.csv.gz --profile quick
```

---

## 4) Core Runtime Flags

### 4.1 Data and IO

- `--config <path>`
- `--target <column>`
- `--delimiter <char>`
- `--output-dir <path>`
- `--profile <auto|quick|thorough|minimal>`
- `--low-memory <true|false>` (enables memory-safe defaults for large datasets)

### 4.2 Parsing Hints

- `--datetime-locale-hint <auto|dmy|mdy>`
- `--numeric-locale-hint <auto|us|eu>`

### 4.2.1 Pre-Flight Data Hygiene (Default)

Before preprocessing and univariate profiling, Seldon now performs a pre-flight sparse-column cull:

- Any column with more than 95% missingness is removed.
- If `--target` is explicitly provided, the target column is protected from this auto-cull.

Missing-token detection also treats common unknown placeholders as missing values, including:

- `unknown`, `unk`, `?`, `-`, `--`, `tbd`

### 4.2.2 Boolean and Multi-Select Categorical Handling (Default)

Seldon automatically profiles categorical columns and can synthesize numeric indicators for:

- boolean-like labels: `yes/no`, `true/false`, `on/off`, `1/0`
- multi-select cells (lists in one field) split by common separators: `,`, `;`, `|`

Generated features are bounded and appended to the dataset for feature selection/modeling.

Report visibility:

- categorical columns are labeled as `categorical(bool-like)` or `categorical(multi-select)` when detected
- multi-select token prevalence appears in `Multi-Select Token Frequencies`

### 4.3 Plot Controls

- `--plots <none|all|univariate|overall|bivariate|comma-list>`
- `--plot-univariate <true|false>`
- `--plot-overall <true|false>`
- `--plot-bivariate <true|false>`
- `--plot-theme <auto|light|dark>`
- `--plot-grid <true|false>`
- `--plot-point-size <number>`
- `--plot-line-width <number>`

### 4.4 Output Format Controls

- `--generate-html <true|false>`
- `--export-preprocessed <none|csv|parquet>`
- `--export-preprocessed-path <path_base>`

---

## 5) Target and Strategy Flags

### 5.1 Target Resolution

- `--target-strategy <auto|quality|max_variance|last_numeric>`

### 5.2 Feature and Pair Strategies

- `--feature-strategy <auto|adaptive|aggressive|lenient>`
- `--bivariate-strategy <auto|balanced|corr_heavy|importance_heavy>`

### 5.3 Neural Strategy

- `--neural-strategy <auto|none|fast|balanced|expressive>`

---

## 6) Explicit Column Type Overrides

Use this when auto type inference needs deterministic control.

CLI form:

- `--type column:numeric`
- `--type column:categorical`
- `--type column:datetime`

Repeatable usage:

```bash
./seldon /data/data.csv \
  --type timestamp:datetime \
  --type segment:categorical \
  --type spend:numeric
```

Config-file form:

- `type.timestamp: datetime`
- `type.segment: categorical`
- `type.spend: numeric`

---

## 7) Imputation Controls

Per-column key pattern:

- `impute.<column_name>: <strategy>`

Allowed values:

- `auto`
- `mean`
- `median`
- `zero`
- `mode`
- `interpolate`

Example:

```yaml
impute.revenue: median
impute.city: mode
impute.event_time: interpolate
```

---

## 8) Outlier Controls

Outlier method:

- `outlier_method: iqr|zscore|modified_zscore|adjusted_boxplot|lof`

Outlier action:

- `outlier_action: flag|remove|cap`

Optional memory saver:

- `store_outlier_flags_in_report: true|false`

Recommended when memory-constrained:

- `store_outlier_flags_in_report: false`

---

## 9) Scaling Controls

Scaling key:

- `scaling: auto|zscore|minmax|none`

This applies to numeric columns during preprocessing.

---

## 10) Feature Engineering Controls

Enable/disable:

- `feature_engineering_enable_poly: true|false`
- `feature_engineering_enable_log: true|false`
- `feature_engineering_enable_ratio_product_discovery: true|false`

Degree and base:

- `feature_engineering_degree: <int>=1+`
- `feature_engineering_max_base: <int>=2+`
- `feature_engineering_max_pairwise_discovery: <int>=2+`

Expansion cap:

- `feature_engineering_max_generated_columns: <int>=16+`

Example:

```yaml
feature_engineering_enable_poly: true
feature_engineering_enable_log: true
feature_engineering_enable_ratio_product_discovery: true
feature_engineering_degree: 3
feature_engineering_max_base: 10
feature_engineering_max_pairwise_discovery: 24
feature_engineering_max_generated_columns: 600
```

---

## 11) Fast Mode Controls

Fast mode can be explicit or inferred by scale.

Controls:

- `fast_mode: true|false`
- `fast_max_bivariate_pairs: <int>`
- `fast_neural_sample_rows: <int>`

Typical quick run:

```bash
./seldon /data/big.csv --profile quick --fast true --plots none
```

---

## 12) Neural Core Controls

Learning and optimization:

- `neural_learning_rate`
- `neural_optimizer`
- `neural_lookahead_fast_optimizer`
- `neural_lookahead_sync_period`
- `neural_lookahead_alpha`
- `gradient_clip_norm`

Normalization toggles:

- `neural_use_batch_norm`
- `neural_batch_norm_momentum`
- `neural_batch_norm_epsilon`
- `neural_use_layer_norm`
- `neural_layer_norm_epsilon`

Scheduler and stopping:

- `neural_lr_decay`
- `neural_lr_plateau_patience`
- `neural_lr_cooldown_epochs`
- `neural_max_lr_reductions`
- `neural_min_learning_rate`
- `neural_use_validation_loss_ema`
- `neural_validation_loss_ema_beta`

---

## 13) Neural Topology and Memory Guards

Topology shape controls:

- `neural_min_layers`
- `neural_max_layers`
- `neural_fixed_layers`
- `neural_fixed_hidden_nodes`
- `neural_max_hidden_nodes`

Safety limits:

- `neural_max_topology_nodes`
- `neural_max_trainable_params`

Categorical expansion guard:

- `neural_max_one_hot_per_column`

These settings prevent accidental large allocations.

---

## 14) Streaming and Multi-Output Controls

Streaming:

- `neural_streaming_mode: true|false`
- `neural_streaming_chunk_rows: <int>=16+`

Multi-output:

- `neural_multi_output: true|false`
- `neural_max_aux_targets: <int>=0`

---

## 15) Explainability and Uncertainty

Explainability modes:

- `neural_explainability: permutation|integrated_gradients|hybrid`

Sampling controls:

- `neural_integrated_grad_steps`
- `neural_uncertainty_samples`
- `neural_ensemble_members`
- `neural_ensemble_probe_rows`
- `neural_ensemble_probe_epochs`
- `neural_ood_enabled`
- `neural_ood_z_threshold`
- `neural_ood_distance_threshold`
- `neural_drift_psi_warning`
- `neural_drift_psi_critical`
- `neural_importance_parallel`
- `neural_importance_max_rows`
- `neural_importance_trials`

Hybrid blend controls:

- `hybrid_explainability_weight_permutation`
- `hybrid_explainability_weight_integrated_gradients`

---

## 16) Plot and Heuristic Tuning Keys

High-level plot toggles:

- `plot_univariate`
- `plot_overall`
- `plot_bivariate_significant`

Plot style:

- `plot_format`
- `plot_theme`
- `plot_grid`
- `plot_width`
- `plot_height`
- `plot_point_size`
- `plot_line_width`

Selected heuristic tuning keys:

- `significance_alpha`
- `outlier_iqr_multiplier`
- `outlier_z_threshold`
- `feature_min_variance`
- `feature_leakage_corr_threshold`
- `feature_missing_q3_offset`
- `feature_missing_floor`
- `feature_missing_ceiling`
- `feature_aggressive_delta`
- `feature_lenient_delta`
- `bivariate_selection_quantile`
- `bivariate_tier3_fallback_aggressiveness`
- `coherence_weight_small_dataset`
- `coherence_weight_regular_dataset`
- `coherence_overfit_penalty_train_ratio`
- `coherence_benchmark_penalty_ratio`
- `coherence_penalty_step`
- `coherence_weight_min`
- `coherence_weight_max`
- `corr_heavy_max_importance_threshold`
- `corr_heavy_concentration_threshold`
- `importance_heavy_max_importance_threshold`
- `importance_heavy_concentration_threshold`
- `numeric_epsilon`
- `beta_fallback_intervals_start`
- `beta_fallback_intervals_max`
- `beta_fallback_tolerance`
- `overall_corr_heatmap_max_columns`
- `ogive_min_points`
- `ogive_min_unique`
- `box_plot_min_points`
- `box_plot_min_iqr`
- `pie_min_categories`
- `pie_max_categories`
- `pie_max_dominance_ratio`
- `facet_min_rows`
- `facet_max_categories`
- `facet_min_category_share`
- `scatter_downsample_threshold`
- `scatter_fit_min_abs_corr`
- `scatter_fit_min_sample_size`
- `residual_plot_min_abs_corr`
- `residual_plot_min_sample_size`
- `category_numeric_distribution_min_rows`
- `category_numeric_distribution_max_pairs`
- `histogram_density_min_sample`
- `parallel_coordinates_min_rows`
- `parallel_coordinates_max_rows`
- `parallel_coordinates_min_dims`
- `parallel_coordinates_max_dims`
- `time_series_trend_min_rows`
- `gantt_auto_enabled`
- `gantt_min_tasks`
- `gantt_max_tasks`
- `gantt_duration_hours_threshold`

---

## 17) Canonical Config Template

```yaml
dataset: /absolute/path/to/data.csv
target: target_column
delimiter: ,

profile: auto
output_dir: /tmp/seldon_output
verbose_analysis: true
generate_html: false

datetime_locale_hint: auto
numeric_locale_hint: auto

type.event_date: datetime
type.segment: categorical

outlier_method: iqr
outlier_action: cap
store_outlier_flags_in_report: false

scaling: auto

feature_engineering_enable_poly: true
feature_engineering_enable_log: true
feature_engineering_enable_ratio_product_discovery: true
feature_engineering_degree: 2
feature_engineering_max_base: 8
feature_engineering_max_pairwise_discovery: 24
feature_engineering_max_generated_columns: 512

target_strategy: auto
feature_strategy: auto
neural_strategy: auto
bivariate_strategy: auto
bivariate_tier3_fallback_aggressiveness: 1.0

fast_mode: false
fast_max_bivariate_pairs: 2500
fast_neural_sample_rows: 25000

neural_optimizer: lookahead
neural_lookahead_fast_optimizer: adam
neural_lookahead_sync_period: 5
neural_lookahead_alpha: 0.5
neural_learning_rate: 0.001
gradient_clip_norm: 5.0

neural_min_layers: 1
neural_max_layers: 3
neural_fixed_layers: 0
neural_fixed_hidden_nodes: 0
neural_max_hidden_nodes: 128

neural_max_one_hot_per_column: 24
neural_max_topology_nodes: 4096
neural_max_trainable_params: 20000000

neural_streaming_mode: false
neural_streaming_chunk_rows: 2048

neural_multi_output: true
neural_max_aux_targets: 2

neural_explainability: hybrid
neural_integrated_grad_steps: 8
neural_uncertainty_samples: 24
neural_ensemble_members: 3
neural_ensemble_probe_rows: 256
neural_ensemble_probe_epochs: 48
neural_ood_enabled: true
neural_ood_z_threshold: 3.5
neural_ood_distance_threshold: 2.5
neural_drift_psi_warning: 0.15
neural_drift_psi_critical: 0.25
neural_importance_parallel: true
neural_importance_max_rows: 1000
neural_importance_trials: 0

hybrid_explainability_weight_permutation: 0.5
hybrid_explainability_weight_integrated_gradients: 0.5

plots: bivariate,overall
plot_theme: auto
plot_grid: true
plot_width: 1280
plot_height: 720
plot_point_size: 0.8
plot_line_width: 2.0

export_preprocessed: none
export_preprocessed_path: /tmp/preprocessed
```

---

## 18) Execution Recipes

### 18.1 Minimal run

```bash
./seldon /data/input.csv
```

### 18.2 No plots run

```bash
./seldon /data/input.csv --plots none
```

### 18.3 Thorough report run

```bash
./seldon /data/input.csv --profile thorough --plots all
```

### 18.4 Memory-aware run

```bash
./seldon /data/input.csv \
  --profile quick \
  --feature-engineering-max-generated-columns 256 \
  --neural-max-one-hot-per-column 16 \
  --neural-importance-max-rows 1000 \
  --plots none
```

### 18.5 Deterministic schema run

```bash
./seldon /data/input.csv \
  --type date:datetime \
  --type category:categorical \
  --type amount:numeric
```

---

## 19) Output Files

Primary markdown outputs:

- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`

`final_analysis.md` now includes a **Causal Inference (Lite) DAG Candidates** section that ranks heuristic directed edges (likely drivers vs likely proxies) using partial-correlation retention and interaction proxy evidence.

Assets:

- `seldon_report_assets/univariate`
- `seldon_report_assets/bivariate`
- `seldon_report_assets/overall`

Optional html outputs:

- `univariate.html`
- `bivariate.html`
- `neural_synthesis.html`
- `final_analysis.html`

---

## 20) Exit and Failure Behavior

Seldon throws explicit exceptions for:

- invalid configuration,
- missing required files,
- unsupported runtime settings,
- malformed datasets,
- unavailable required converters for chosen input.

If the binary exits non-zero,
inspect the terminal message first,
then verify converter and tool availability.

---

## 21) Troubleshooting Checklist

### 21.1 Build Issues

- verify CMake version,
- verify C++17 compiler,
- rebuild in clean `build/` directory.

### 21.2 XLSX/XLS Issues

- verify `xlsx2csv` and `xls2csv` in `PATH`.

### 21.3 Compressed Input Issues

- verify `gzip` and `unzip` in `PATH`.

### 21.4 Slow Runs

- use `profile quick`,
- set `plots none`,
- reduce explainability rows/trials,
- lower feature engineering cap.

### 21.5 Memory Pressure

- lower one-hot cap,
- lower engineered column cap,
- lower neural topology limits,
- disable outlier flag storage.

### 21.6 Type Misclassification

- apply explicit `--type` overrides,
- or use `type.<column>` in config.

### 21.7 Garbled Terminal Output

- prefer `verbose_analysis=true` for full logs,
- spinner is suppressed in verbose mode.

---

## 22) Practical Verification Commands

Check binary help behavior quickly:

```bash
./seldon --interactive
```

Check key dependencies:

```bash
command -v gnuplot
command -v pandoc
command -v xlsx2csv
command -v xls2csv
command -v gzip
command -v unzip
```

Run with explicit output directory:

```bash
./seldon /data/input.csv --output-dir /tmp/seldon_output
```

---

## 23) Quick Reference Lists

### 23.1 Boolean-like keys you will frequently use

- `fast_mode`
- `plot_univariate`
- `plot_overall`
- `plot_bivariate_significant`
- `generate_html`
- `verbose_analysis`
- `neural_use_batch_norm`
- `neural_use_layer_norm`
- `neural_use_validation_loss_ema`
- `neural_streaming_mode`
- `neural_multi_output`
- `neural_importance_parallel`
- `feature_engineering_enable_poly`
- `feature_engineering_enable_log`
- `store_outlier_flags_in_report`

### 23.2 High-impact integer keys

- `kfold`
- `fast_max_bivariate_pairs`
- `fast_neural_sample_rows`
- `neural_min_layers`
- `neural_max_layers`
- `neural_fixed_layers`
- `neural_fixed_hidden_nodes`
- `neural_max_hidden_nodes`
- `neural_streaming_chunk_rows`
- `neural_max_aux_targets`
- `neural_integrated_grad_steps`
- `neural_uncertainty_samples`
- `neural_importance_max_rows`
- `neural_importance_trials`
- `neural_max_one_hot_per_column`
- `neural_max_topology_nodes`
- `neural_max_trainable_params`
- `feature_engineering_degree`
- `feature_engineering_max_base`
- `feature_engineering_max_generated_columns`

### 23.3 High-impact floating-point keys

- `neural_learning_rate`
- `gradient_clip_norm`
- `neural_lookahead_alpha`
- `neural_batch_norm_momentum`
- `neural_batch_norm_epsilon`
- `neural_layer_norm_epsilon`
- `neural_lr_decay`
- `neural_min_learning_rate`
- `neural_validation_loss_ema_beta`
- `neural_categorical_input_l2_boost`
- `hybrid_explainability_weight_permutation`
- `hybrid_explainability_weight_integrated_gradients`

---

## 24) Related Documentation

- [README](../README.md)
- [ARCHITECTURE](ARCHITECTURE.md)
