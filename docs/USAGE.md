# Usage Guide

This guide covers build, run, configuration, and operational patterns for Seldon.

## Build

From project root:

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

Disable OpenMP if needed:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
cmake --build . -j
```

## Run

From the build directory:

```bash
./seldon /path/to/data.csv
```

With config file:

```bash
./seldon /path/to/data.csv --config /path/to/config.yaml
```

## Common CLI Patterns

### Target and delimiter

```bash
./seldon /path/to/data.csv --target sales --delimiter ';'
```

### Plot control

```bash
./seldon /path/to/data.csv --plots bivariate
./seldon /path/to/data.csv --plots all
./seldon /path/to/data.csv --plot-univariate true --plot-overall false --plot-bivariate true
```

### Strategy control

```bash
./seldon /path/to/data.csv \
  --target-strategy auto \
  --feature-strategy auto \
  --neural-strategy auto \
  --bivariate-strategy auto
```

### Fast mode (large datasets)

```bash
./seldon /path/to/data.csv --fast true --fast-max-bivariate-pairs 2500 --fast-neural-sample-rows 25000
```

### Reproducibility and stability

```bash
./seldon /path/to/data.csv --neural-seed 1337 --benchmark-seed 1337 --gradient-clip 5.0
```

## Config File Format

Seldon accepts lightweight YAML/JSON-like `key: value` files.

### Minimal example

```yaml
dataset: /path/to/data.csv
target: sales
outlier_method: iqr
outlier_action: flag
scaling: auto
kfold: 5
```

### Practical extended example

```yaml
report: neural_synthesis.md
assets_dir: seldon_report_assets
delimiter: ,

# Plot settings
plot_format: png
plot_width: 1280
plot_height: 720
plot_univariate: false
plot_overall: false
plot_bivariate_significant: true
plots: bivariate

# Runtime toggles
generate_html: false
verbose_analysis: true

# Seeds and clipping
neural_seed: 1337
benchmark_seed: 1337
gradient_clip_norm: 5.0

# Core modes
target_strategy: auto
feature_strategy: auto
neural_strategy: auto
bivariate_strategy: auto

# Fast mode
fast_mode: false
fast_max_bivariate_pairs: 2500
fast_neural_sample_rows: 25000

# Statistics and thresholds
significance_alpha: 0.05
outlier_iqr_multiplier: 1.5
outlier_z_threshold: 3.0
feature_min_variance: 1e-10
feature_leakage_corr_threshold: 0.995
max_feature_missing_ratio: -1

# Numeric runtime controls
numeric_epsilon: 1e-12
beta_fallback_intervals_start: 4096
beta_fallback_intervals_max: 65536
beta_fallback_tolerance: 1e-8
overall_corr_heatmap_max_columns: 50

# Optional column rules
exclude: id,notes
impute.sales: median
impute.region: mode
```

## Key Options

### Data and preprocessing

- `dataset`
- `target`
- `delimiter`
- `outlier_method`: `iqr` | `zscore` | `modified_zscore` | `adjusted_boxplot` | `lof`
- `outlier_action`: `flag` | `remove` | `cap`
- `scaling`: `auto` | `zscore` | `minmax` | `none`
- `max_feature_missing_ratio`: `-1` or `[0,1]`
- `impute.<column>`: `auto` | `mean` | `median` | `zero` | `mode` | `interpolate`

### Pipeline strategies

- `target_strategy`: `auto` | `quality` | `max_variance` | `last_numeric`
- `feature_strategy`: `auto` | `adaptive` | `aggressive` | `lenient`
- `neural_strategy`: `auto` | `none` | `fast` | `balanced` | `expressive`
- `bivariate_strategy`: `auto` | `balanced` | `corr_heavy` | `importance_heavy`
- `bivariate_selection_quantile`: `-1` or `[0,1]`

### Plot and output

- `plots`: `none` | `bivariate` | `univariate` | `overall` | `all`
- when no plot aliases/flags are provided, Seldon auto-enables a dynamic plot set for the current dataset (avoids under-plotting and suppresses low-value plot families on tiny datasets)
- plot generation auto-selects suitable visuals by data shape (adaptive histogram with KDE, scatter with confidence/residual diagnostics, faceted scatter, stacked bivariate profiles, clustered heatmap, parallel coordinates, Q-Q plots, pie/bar fallback, and project-like Gantt)
- `plot_univariate`
- `plot_overall`
- `plot_bivariate_significant`
- `plot_format`: `png` | `svg` | `pdf`
- `plot_theme`: `auto` | `light` | `dark`
- `plot_grid`: `true` | `false`
- `plot_point_size`, `plot_line_width`
- `plot_width`, `plot_height`
- `ogive_min_points`, `ogive_min_unique`
- `box_plot_min_points`, `box_plot_min_iqr`
- `pie_min_categories`, `pie_max_categories`, `pie_max_dominance_ratio`
- `scatter_fit_min_abs_corr`, `scatter_fit_min_sample_size`
- `gantt_auto_enabled`, `gantt_min_tasks`, `gantt_max_tasks`, `gantt_duration_hours_threshold`
- `report`, `assets_dir`
- `generate_html`

### Neural controls

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

### Numeric robustness

- `significance_alpha`
- `numeric_epsilon`
- `beta_fallback_intervals_start`
- `beta_fallback_intervals_max`
- `beta_fallback_tolerance`
- `overall_corr_heatmap_max_columns`

## Outputs

Always generated:

- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`

Generated when plotting is enabled and `gnuplot` is available:

- `seldon_report_assets/univariate/`
- `seldon_report_assets/bivariate/`
- `seldon_report_assets/overall/`

Generated when `generate_html=true` and `pandoc` is available:

- `univariate.html`
- `bivariate.html`
- `neural_synthesis.html`
- `final_analysis.html`

## Runtime Notes

- Outlier detection runs before imputation using observed numeric values only.
- Correlation heatmap computation is capped by `overall_corr_heatmap_max_columns` to keep wide datasets tractable.
- Feature-importance evaluation uses adaptive row sampling and trial limiting for large data.
- Numeric parsing supports configurable separator policy for locale-like formats.
- CSV load is streaming two-pass and avoids buffering full datasets in memory.
- Automated EDA sections include contingency (chi-square/Cramér’s V/odds ratio), one-way ANOVA with post-hoc summary, PCA explained variance + projection plots, k-means profile with silhouette/gap summary, bootstrap confidence intervals, missingness matrix/correlation, and regression diagnostics (residual/Q-Q/Cook/VIF).

## Troubleshooting

### Build fails with OpenMP flags

- Reconfigure with OpenMP disabled:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
```

### No charts generated

- Ensure `gnuplot` is installed and in PATH.
- Reports still generate without charts.

### HTML files not generated

- Ensure `pandoc` is installed.
- Set `generate_html: true`.

### Config parse errors

- Verify one `key: value` per line.
- Avoid malformed quotes.
- Check unsupported option names.

## Related Docs

- [README.md](../README.md)
- [docs/ARCHITECTURE.md](ARCHITECTURE.md)
