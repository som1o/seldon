# Usage Reference

## 1. Build and Installation

### 1.1 Prerequisites

- C++17 compiler
- CMake 3.16+
- Optional: OpenMP runtime
- Optional: `gnuplot` (for plots)
- Optional: `pandoc` (for HTML export)

### 1.2 Build Commands

From project root:

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

OpenMP-disabled build:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
cmake --build . -j
```

The produced executable is `seldon` in the build directory.

## 2. Execution Model

### 2.1 Minimal Invocation

```bash
./seldon /path/to/data.csv
```

### 2.2 Invocation with Configuration File

```bash
./seldon /path/to/data.csv --config /path/to/config.yaml
```

### 2.3 Representative Invocation Patterns

Target and delimiter:

```bash
./seldon /path/to/data.csv --target sales --delimiter ';'
```

Plot selection:

```bash
./seldon /path/to/data.csv --plots bivariate
./seldon /path/to/data.csv --plots all
```

Strategy selection:

```bash
./seldon /path/to/data.csv \
  --target-strategy auto \
  --feature-strategy auto \
  --neural-strategy auto \
  --bivariate-strategy auto
```

Large-data runtime controls:

```bash
./seldon /path/to/data.csv --fast true --fast-max-bivariate-pairs 2500 --fast-neural-sample-rows 25000
```

Reproducibility controls:

```bash
./seldon /path/to/data.csv --neural-seed 1337 --benchmark-seed 1337 --gradient-clip 5.0
```

## 3. Configuration File Specification

Seldon accepts a lightweight `key: value` configuration format. Values may be scalar strings, numbers, or booleans, depending on option semantics.

### 3.1 Minimal Example

```yaml
dataset: /path/to/data.csv
target: sales
outlier_method: iqr
outlier_action: flag
scaling: auto
kfold: 5
```

### 3.2 Extended Example

```yaml
report: neural_synthesis.md
assets_dir: seldon_report_assets
delimiter: ,

plot_format: png
plot_width: 1280
plot_height: 720
plot_univariate: false
plot_overall: false
plot_bivariate_significant: true
plots: bivariate

generate_html: false
verbose_analysis: true

neural_seed: 1337
benchmark_seed: 1337
gradient_clip_norm: 5.0

target_strategy: auto
feature_strategy: auto
neural_strategy: auto
bivariate_strategy: auto

fast_mode: false
fast_max_bivariate_pairs: 2500
fast_neural_sample_rows: 25000

significance_alpha: 0.05
outlier_iqr_multiplier: 1.5
outlier_z_threshold: 3.0
feature_min_variance: 1e-10
feature_leakage_corr_threshold: 0.995
max_feature_missing_ratio: -1

numeric_epsilon: 1e-12
beta_fallback_intervals_start: 4096
beta_fallback_intervals_max: 65536
beta_fallback_tolerance: 1e-8
overall_corr_heatmap_max_columns: 50

exclude: id,notes
impute.sales: median
impute.region: mode
```

## 4. Option Reference

### 4.1 Data and Preprocessing

- `dataset`, `target`, `delimiter`
- `outlier_method`: `iqr` | `zscore` | `modified_zscore` | `adjusted_boxplot` | `lof`
- `outlier_action`: `flag` | `remove` | `cap`
- `scaling`: `auto` | `zscore` | `minmax` | `none`
- `max_feature_missing_ratio`: `-1` or value in $[0,1]$
- `impute.<column>`: `auto` | `mean` | `median` | `zero` | `mode` | `interpolate`

### 4.2 Strategy Parameters

- `target_strategy`: `auto` | `quality` | `max_variance` | `last_numeric`
- `feature_strategy`: `auto` | `adaptive` | `aggressive` | `lenient`
- `neural_strategy`: `auto` | `none` | `fast` | `balanced` | `expressive`
- `bivariate_strategy`: `auto` | `balanced` | `corr_heavy` | `importance_heavy`
- `bivariate_selection_quantile`: `-1` or value in $[0,1]$

### 4.3 Plot and Output Controls

- `plots`: `none` | `bivariate` | `univariate` | `overall` | `all`
- `plot_univariate`, `plot_overall`, `plot_bivariate_significant`
- `plot_format`: `png` | `svg` | `pdf`
- `plot_theme`: `auto` | `light` | `dark`
- `plot_grid`, `plot_point_size`, `plot_line_width`
- `plot_width`, `plot_height`
- `ogive_min_points`, `ogive_min_unique`
- `box_plot_min_points`, `box_plot_min_iqr`
- `pie_min_categories`, `pie_max_categories`, `pie_max_dominance_ratio`
- `scatter_fit_min_abs_corr`, `scatter_fit_min_sample_size`
- `gantt_auto_enabled`, `gantt_min_tasks`, `gantt_max_tasks`, `gantt_duration_hours_threshold`
- `report`, `assets_dir`, `generate_html`

### 4.4 Neural and Optimization Controls

- `neural_seed`, `benchmark_seed`
- `gradient_clip_norm`
- `neural_optimizer`: `sgd` | `adam` | `lookahead`
- `neural_lookahead_fast_optimizer`: `sgd` | `adam`
- `neural_lookahead_sync_period`, `neural_lookahead_alpha`
- `neural_use_batch_norm`, `neural_batch_norm_momentum`, `neural_batch_norm_epsilon`
- `neural_use_layer_norm`, `neural_layer_norm_epsilon`
- `neural_lr_decay`, `neural_lr_plateau_patience`, `neural_lr_cooldown_epochs`
- `neural_max_lr_reductions`, `neural_min_learning_rate`
- `neural_use_validation_loss_ema`, `neural_validation_loss_ema_beta`
- `neural_categorical_input_l2_boost`

### 4.5 Numerical Robustness Controls

- `significance_alpha`
- `numeric_epsilon`
- `beta_fallback_intervals_start`
- `beta_fallback_intervals_max`
- `beta_fallback_tolerance`
- `overall_corr_heatmap_max_columns`

## 5. Output Specification

### 5.1 Always Generated

- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`

### 5.2 Conditionally Generated

When plotting is enabled and `gnuplot` is available:

- `seldon_report_assets/univariate/`
- `seldon_report_assets/bivariate/`
- `seldon_report_assets/overall/`

When `generate_html=true` and `pandoc` is available:

- `univariate.html`
- `bivariate.html`
- `neural_synthesis.html`
- `final_analysis.html`

## 6. Operational Notes

- Outlier detection is computed on observed numeric values before imputation.
- Correlation heatmap computation is bounded by `overall_corr_heatmap_max_columns`.
- Large-data execution can be constrained through fast-mode pair and sample limits.
- CSV ingestion follows a streaming two-pass model rather than full-file buffering.

## 7. Troubleshooting

### 7.1 OpenMP Build Errors

Reconfigure with OpenMP disabled:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
cmake --build . -j
```

### 7.2 Missing Plot Outputs

Ensure `gnuplot` is available in `PATH` and plotting is enabled by configuration.

### 7.3 Missing HTML Outputs

Ensure `pandoc` is available in `PATH` and set `generate_html: true`.

### 7.4 Configuration Parse Failures

Use one `key: value` pair per line and verify option names and quoting.

## 8. Cross-Reference

- [Project Overview](../README.md)
- [Architecture Reference](ARCHITECTURE.md)
