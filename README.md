# Seldon

Seldon is a C++ automated analytics engine that performs end-to-end data processing with minimal user input:

- Typed CSV ingestion (numeric, categorical, datetime)
- Automated preprocessing (missing values, outliers, scaling)
- Automated EDA plotting via gnuplot
- Baseline model benchmarking (linear, ridge, tree-stump)
- Neural-network training with sensible defaults
- Unified report generation

## Build

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

To force single-threaded builds:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
```

## Run

```bash
./seldon /path/to/data.csv
```

Optional:

```bash
./seldon /path/to/data.csv --config config.yaml
./seldon /path/to/data.csv --target sales --delimiter ';'
./seldon /path/to/data.csv --plots bivariate,univariate,overall
./seldon /path/to/data.csv --neural-seed 1337 --gradient-clip 5.0
./seldon /path/to/data.csv --max-feature-missing-ratio -1
./seldon /path/to/data.csv --target-strategy auto --feature-strategy auto --neural-strategy auto --bivariate-strategy auto
./seldon /path/to/data.csv --feature-min-variance 1e-9 --feature-leakage-corr-threshold 0.99 --bivariate-selection-quantile 0.70
./seldon /path/to/data.csv --fast true --fast-max-bivariate-pairs 2500 --fast-neural-sample-rows 25000
```

Extended documentation:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/USAGE.md](docs/USAGE.md)

## Architecture

### Entry and Orchestration
- `src/main.cpp`: Minimal entrypoint
- `src/AutomationPipeline.cpp`: Full end-to-end pipeline orchestration

### Ingestion and Typing
- `src/TypedDataset.cpp` / `include/TypedDataset.h`
  - Robust CSV parsing (quoted fields, escaped quotes, multiline quoted text)
  - Configurable delimiter
  - Type inference into typed column storage:
    - numeric (`std::vector<double>`)
    - categorical (`std::vector<std::string>`)
    - datetime (`std::vector<int64_t>`)

### Configuration
- `src/AutoConfig.cpp` / `include/AutoConfig.h`
  - CLI + file-based config resolution
  - Validated config values with explicit configuration exceptions

### Preprocessing
- `src/Preprocessor.cpp` / `include/Preprocessor.h`
  - Missing handling:
    - numeric: mean/median/zero/interpolation
    - categorical: mode
    - datetime: interpolation/fill
  - Outlier processing:
    - methods: `iqr`, `zscore`
    - actions: `flag`, `remove`, `cap`
  - Feature scaling:
    - `auto`, `zscore`, `minmax`, `none`
  - Captures preprocessing report metadata

### Modeling and Benchmarking
- `src/BenchmarkEngine.cpp` / `include/BenchmarkEngine.h`
  - Automated k-fold evaluation for:
    - linear regression
    - ridge regression
    - decision tree stump baseline
  - Dynamic benchmark ranking by task objective
  - Uses shared `MathUtils` regression solver (ridge via augmented rows)
  - Reports RMSE, R2, and binary accuracy (when applicable)

- `src/NeuralNet.cpp` / `include/NeuralNet.h`
  - Dense feed-forward NN with ADAM/SGD options
  - Data-adaptive auto defaults (epochs, batch size, hidden width, dropout, patience)
  - Auto output/loss selection by inferred task type (classification vs regression)
  - Optional bypass via `neural_strategy: none`
  - Binary model serialization with integrity checks

### Dynamic Decision Layer
- `src/AutomationPipeline.cpp`
  - Auto target selection when `--target` is not provided
  - Strategy registry for target/feature/neural/bivariate policies
  - Auto task inference from target distribution (binary vs regression)
  - Adaptive sparse-feature gating with configurable threshold
  - Coherence-aware neural relevance scoring for bivariate selection
  - Auto decision log emitted into `neural_synthesis.md`

### Numerical Core
- `src/MathUtils.cpp` / `include/MathUtils.h`
  - Cache-friendlier matrix multiplication (`A * B^T` row inner-products, OpenMP-aware)
  - Statistical significance utilities with guarded incomplete-beta fallback path and adaptive numerical controls
  - Contract-style API documentation for preconditions/postconditions/exceptions

### Plotting and Reporting
- `src/GnuplotEngine.cpp` / `include/GnuplotEngine.h`
  - Generates PNG plots in dedicated folders (`univariate`, `bivariate`, `overall`)
  - Safe id/path handling and script/value escaping
  - Automatically removes intermediate `.dat` / `.plt` files

- `src/ReportEngine.cpp` / `include/ReportEngine.h`
  - Markdown report writer with section/table/image formatting
  - Image links normalized to portable relative paths when possible

## Output

By default, each run generates:

- `univariate.md` (univariate summary)
- `bivariate.md` (all pair combinations + final significant table)
- `neural_synthesis.md` (detailed neural lattice training trace + synthesis)
- `final_analysis.md` (significant findings only, selected by neural decision engine)
- `seldon_report_assets/bivariate/` (PNG plots for selected significant bivariate results)

When supervised plotting is explicitly enabled, it also generates:

- `seldon_report_assets/univariate/` (univariate histograms/category distributions)
- `seldon_report_assets/overall/` (overall missingness/heatmap/loss trend plots)

## Config File (YAML/JSON-like key:value)

Example:

```yaml
target: sales
report: neural_synthesis.md
assets_dir: seldon_report_assets
delimiter: ,
outlier_method: iqr
outlier_action: flag
scaling: auto
kfold: 5
plot_format: png
plot_width: 1280
plot_height: 720
plot_univariate: false
plot_overall: false
plot_bivariate_significant: true
generate_html: false
verbose_analysis: true
neural_seed: 1337
benchmark_seed: 1337
gradient_clip_norm: 5.0
significance_alpha: 0.05
outlier_iqr_multiplier: 1.5
outlier_z_threshold: 3.0
max_feature_missing_ratio: -1
target_strategy: auto
feature_strategy: auto
neural_strategy: auto
bivariate_strategy: auto
fast_mode: false
fast_max_bivariate_pairs: 2500
fast_neural_sample_rows: 25000
coherence_weight_small_dataset: 0.55
coherence_weight_regular_dataset: 0.70
coherence_overfit_penalty_train_ratio: 1.5
coherence_benchmark_penalty_ratio: 1.5
coherence_penalty_step: 0.20
coherence_weight_min: 0.20
coherence_weight_max: 0.85
corr_heavy_max_importance_threshold: 0.65
corr_heavy_concentration_threshold: 0.55
importance_heavy_max_importance_threshold: 0.30
importance_heavy_concentration_threshold: 0.40
numeric_epsilon: 1e-12
beta_fallback_intervals_start: 4096
beta_fallback_intervals_max: 65536
beta_fallback_tolerance: 1e-8
plots: bivariate
exclude: id,notes
impute.sales: median
impute.region: mode
feature_min_variance: 1e-10
feature_leakage_corr_threshold: 0.995
feature_missing_q3_offset: 0.15
feature_missing_floor: 0.35
feature_missing_ceiling: 0.95
feature_aggressive_delta: 0.20
feature_lenient_delta: 0.20
bivariate_selection_quantile: -1
```

Supported scalar keys:

- `dataset`
- `target`
- `report`
- `assets_dir`
- `delimiter`
- `outlier_method` (`iqr`|`zscore`)
- `outlier_action` (`flag`|`remove`|`cap`)
- `scaling` (`auto`|`zscore`|`minmax`|`none`)
- `kfold`
- `plot_format` (`png`|`svg`|`pdf`)
- `plot_width`
- `plot_height`
- `plot_univariate` (`true`|`false`)
- `plot_overall` (`true`|`false`)
- `plot_bivariate_significant` (`true`|`false`)
- `verbose_analysis` (`true`|`false`)
- `neural_seed` (unsigned integer)
- `benchmark_seed` (unsigned integer)
- `gradient_clip_norm` (>= 0)
- `significance_alpha` (`(0,1)`)
- `outlier_iqr_multiplier` (> 0)
- `outlier_z_threshold` (> 0)
- `max_feature_missing_ratio` (`-1` for auto, or fixed `[0,1]`)
- `target_strategy` (`auto`|`quality`|`max_variance`|`last_numeric`)
- `feature_strategy` (`auto`|`adaptive`|`aggressive`|`lenient`)
- `neural_strategy` (`auto`|`none`|`fast`|`balanced`|`expressive`)
- `bivariate_strategy` (`auto`|`balanced`|`corr_heavy`|`importance_heavy`)
- `fast_mode` (`true`|`false`)
- `fast_max_bivariate_pairs` (> 0)
- `fast_neural_sample_rows` (> 0)
- `generate_html` (`true`|`false`, enables self-contained HTML via `pandoc`)
- `plots` (comma-separated profile: `none`, `bivariate`, `univariate`, `overall`, `all`)
- `exclude` (comma-separated)
- `impute.<columnName>`
- `feature_min_variance` (>= 0)
- `feature_leakage_corr_threshold` (`[0,1]`)
- `feature_missing_q3_offset` (>= 0)
- `feature_missing_floor` (`[0,1]`)
- `feature_missing_ceiling` (`[0,1]`, must be >= floor)
- `feature_aggressive_delta` (>= 0)
- `feature_lenient_delta` (>= 0)
- `bivariate_selection_quantile` (`-1` to keep strategy default, or `[0,1]`)
- `coherence_weight_*`, `coherence_*` (coherent-importance blending controls)
- `corr_heavy_*`, `importance_heavy_*` (auto bivariate policy triggers)
- `numeric_epsilon` (> 0)
- `beta_fallback_intervals_start`, `beta_fallback_intervals_max`, `beta_fallback_tolerance`

### Heuristic Notes

- `feature_strategy=adaptive` uses `Q3(missing_ratio)+feature_missing_q3_offset`, clamped by floor/ceiling.
- `feature_strategy=aggressive` shifts that adaptive threshold downward; `lenient` shifts upward.
- `neural_strategy=none` disables NN training and uses correlation-driven relevance.
- `fast_mode=true` forces `neural_strategy=fast`, caps bivariate pair search, and samples rows for neural training on huge datasets.
- `bivariate_strategy=auto` chooses between balanced/corr-heavy/importance-heavy based on feature-importance concentration.

## Robustness Notes

- Configuration values are validated early and fail fast with contextual errors, including config-file line numbers.
- Report generation escapes HTML content to avoid malformed pages.
- Gnuplot scripts/ids are sanitized and plotting gracefully degrades if gnuplot is unavailable.
- Typed ingestion preserves non-numeric information for downstream analytics.
- Incomplete beta evaluation uses a continued-fraction primary path and a deterministic adaptive midpoint fallback; fallback accuracy/speed is tunable via `beta_fallback_*` keys.
- Benchmark linear/ridge regression now routes through shared `MathUtils` regression routines for better numerical stability than ad-hoc elimination.
- Gnuplot invocation uses `std::system` in a local-tool context, with identifier sanitization and quoting/escaping to reduce command/script injection risk.
- Missing-value masks use `std::vector<uint8_t>` for predictable semantics and efficient scans.
- Datetime parsing supports multiple common forms: `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`, `MM/DD/YYYY`, and `DD-MM-YYYY`.
- If a detected datetime column has too many unparseable non-missing values (mixed/unexpected formats), Seldon safely falls back to categorical for that column.

## Current Scope

Seldon intentionally favors a strong automated foundation over advanced model complexity:

- Decision tree baseline currently uses a stump for speed and robustness.
- Neural optional mode (`neural_strategy=none`) enables lightweight runs on very large datasets.
- Config parser supports practical YAML/JSON-like key:value files, not full spec-complete parsing.
- `TypedDataset` is the production data path and orchestration targets full automation and dynamic model/feature selection.
- Matrix inversion via Gaussian elimination is $O(n^3)$; for typical dataset sizes this is sufficient, while very large-scale use may benefit from a more numerically stable solver.
- The neural network implementation is intentionally feature-rich but currently monolithic; future modularization (for example, introducing dedicated layer abstractions) would improve long-term maintainability.

## License

MIT (see `LICENSE`).
