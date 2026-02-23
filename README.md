# Seldon

Seldon is a C++ automated analytics engine for CSV datasets.

It performs typed ingestion, preprocessing, baseline benchmarking, neural relevance analysis, and report generation with minimal manual setup.

## What Seldon Does

- Typed CSV ingestion (`numeric`, `categorical`, `datetime`)
- Missing-value handling and outlier handling
- Automatic feature scaling
- Baseline model benchmarking (linear, ridge, tree-stump)
- Feed-forward neural analysis for feature relevance
- Bivariate significance + neural-aware selection
- Markdown reports and optional plot assets

## Requirements

- C++17 compiler (GCC/Clang)
- CMake 3.16+
- Optional: OpenMP (for parallel sections)
- Optional: `gnuplot` (for charts)
- Optional: `pandoc` (for HTML report export)

## Build

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

Disable OpenMP:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
```

## Quick Start

```bash
./seldon /path/to/data.csv
```

With config file:

```bash
./seldon /path/to/data.csv --config config.yaml
```

Common options:

```bash
./seldon /path/to/data.csv --target sales --delimiter ';'
./seldon /path/to/data.csv --plots bivariate,univariate,overall
./seldon /path/to/data.csv --fast true --fast-max-bivariate-pairs 2500 --fast-neural-sample-rows 25000
./seldon /path/to/data.csv --target-strategy auto --feature-strategy auto --neural-strategy auto --bivariate-strategy auto
```

## Reports and Assets

Default outputs:

- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`

Plot assets (when plotting is enabled and `gnuplot` exists):

- `seldon_report_assets/univariate/`
- `seldon_report_assets/bivariate/`
- `seldon_report_assets/overall/`

HTML output (when `generate_html=true` and `pandoc` exists):

- `univariate.html`
- `bivariate.html`
- `neural_synthesis.html`
- `final_analysis.html`

## Configuration

Seldon supports CLI overrides and a lightweight YAML/JSON-like `key: value` config.

### Minimal example

```yaml
dataset: /path/to/data.csv
target: sales
outlier_method: iqr
outlier_action: flag
scaling: auto
kfold: 5
plots: bivariate
```

### Extended example

```yaml
report: neural_synthesis.md
assets_dir: seldon_report_assets
delimiter: ,

# Plot control
plot_format: png
plot_width: 1280
plot_height: 720
plot_univariate: false
plot_overall: false
plot_bivariate_significant: true
plots: bivariate

# Runtime behavior
generate_html: false
verbose_analysis: true

# Seeds and stability
neural_seed: 1337
benchmark_seed: 1337
gradient_clip_norm: 5.0

# Strategy controls
target_strategy: auto
feature_strategy: auto
neural_strategy: auto
bivariate_strategy: auto

# Fast mode controls
fast_mode: false
fast_max_bivariate_pairs: 2500
fast_neural_sample_rows: 25000

# Statistical / numeric tuning
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

# Optional per-column rules
exclude: id,notes
impute.sales: median
impute.region: mode
```

### Key options (summary)

- `outlier_method`: `iqr` | `zscore`
- `outlier_action`: `flag` | `remove` | `cap`
- `scaling`: `auto` | `zscore` | `minmax` | `none`
- `target_strategy`: `auto` | `quality` | `max_variance` | `last_numeric`
- `feature_strategy`: `auto` | `adaptive` | `aggressive` | `lenient`
- `neural_strategy`: `auto` | `none` | `fast` | `balanced` | `expressive`
- `bivariate_strategy`: `auto` | `balanced` | `corr_heavy` | `importance_heavy`
- `plots`: `none` | `bivariate` | `univariate` | `overall` | `all`
- `plots` now auto-selects suitable chart families: histogram, scatter, heatmap, ogive, box plot, pie chart, and project-timeline Gantt (when timeline-like columns exist)
- `plot_theme`: `auto` | `light` | `dark`
- `plot_grid`: `true|false`, `plot_point_size`, `plot_line_width`
- Suitability knobs: `ogive_min_points`, `ogive_min_unique`, `box_plot_min_points`, `box_plot_min_iqr`, `pie_min_categories`, `pie_max_categories`, `pie_max_dominance_ratio`
- Fit-line knobs: `scatter_fit_min_abs_corr`, `scatter_fit_min_sample_size`
- Gantt knobs: `gantt_auto_enabled`, `gantt_min_tasks`, `gantt_max_tasks`, `gantt_duration_hours_threshold`
- `overall_corr_heatmap_max_columns`: limits correlation heatmap size

## Notes on Current Behavior

- Outlier detection is performed on observed numeric values before imputation.
- Correlation heatmap work is capped by `overall_corr_heatmap_max_columns` to avoid $O(n^2)$ blowups on very wide datasets.
- Feature-importance evaluation uses adaptive sampling/trials on large datasets for runtime control.
- Numeric parsing supports configurable separator handling for locale-like formats.
- CSV loading is streaming/two-pass and does not keep the entire file buffered in memory.

## Project Layout

- `src/main.cpp` — entry point
- `src/AutomationPipeline.cpp` — orchestration
- `src/TypedDataset.cpp` — typed ingestion
- `src/Preprocessor.cpp` — preprocessing
- `src/BenchmarkEngine.cpp` — baseline models
- `src/NeuralNet.cpp` / `src/NeuralLayer.cpp` — neural analysis
- `src/MathUtils.cpp` — statistical and matrix utilities
- `src/ReportEngine.cpp` — markdown reporting
- `src/GnuplotEngine.cpp` — optional plotting backend

## Documentation

- [docs/USAGE.md](docs/USAGE.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## License

MIT (see [LICENSE](LICENSE)).
