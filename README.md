# Seldon

Seldon is a C analytical pipeline for structured CSV data. It performs typed ingestion, preprocessing, statistical analysis, baseline predictive benchmarking, neural relevance analysis, and report synthesis in a single execution flow.

The system is designed for reproducible exploratory and diagnostic analysis on tabular datasets with mixed variable types (`numeric`, `categorical`, and `datetime`).

## Scope

Seldon executes the following stages:

1. CSV ingestion with per-column type inference and row-aligned storage.
2. Data preprocessing (missingness treatment, outlier handling, scaling, and derived-feature construction).
3. Univariate and bivariate statistical analysis.
4. Baseline model benchmarking with fold-based evaluation.
5. Feed-forward neural analysis for feature relevance estimation.
6. Report generation in Markdown, with optional plots and HTML conversion.

## System Requirements

- C++17-compatible compiler (`g++` or `clang++`)
- CMake 3.16 or newer
- Optional: OpenMP runtime for parallel execution
- Optional: `gnuplot` for figure generation
- Optional: `pandoc` for HTML report export

## Build Procedure

From the project root:

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

Build without OpenMP:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
cmake --build . -j
```

## Execution

Basic execution:

```bash
./seldon /path/to/data.csv
```

Execution with explicit configuration:

```bash
./seldon /path/to/data.csv --config /path/to/config.yaml
```

Representative command forms:

```bash
./seldon /path/to/data.csv --target sales --delimiter ';'
./seldon /path/to/data.csv --plots bivariate
./seldon /path/to/data.csv --fast true --fast-max-bivariate-pairs 2500 --fast-neural-sample-rows 25000
./seldon /path/to/data.csv --target-strategy auto --feature-strategy auto --neural-strategy auto --bivariate-strategy auto
```

## Output Artifacts

Default report files:

- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`

Plot asset directories (generated when plotting is enabled and `gnuplot` is available):

- `seldon_report_assets/univariate/`
- `seldon_report_assets/bivariate/`
- `seldon_report_assets/overall/`

HTML report files (generated when `generate_html=true` and `pandoc` is available):

- `univariate.html`
- `bivariate.html`
- `neural_synthesis.html`
- `final_analysis.html`

## Configuration Interface

Seldon accepts runtime parameters from command-line options and from a lightweight `key: value` configuration file. Command-line values override file-defined values.

Minimal configuration example:

```yaml
dataset: /path/to/data.csv
target: sales
outlier_method: iqr
outlier_action: flag
scaling: auto
kfold: 5
plots: bivariate
```

Extended configuration example:

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

## Documentation Index

- [Usage Reference](docs/USAGE.md)
- [Architecture Reference](docs/ARCHITECTURE.md)

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
