# Seldon

Seldon is a C++ automated analytics engine that performs end-to-end data processing with minimal user input:

- Typed CSV ingestion (numeric, categorical, datetime)
- Automated preprocessing (missing values, outliers, scaling)
- Automated EDA plotting via gnuplot
- Baseline model benchmarking (linear, ridge, tree-stump)
- Neural-network training with sensible defaults
- Unified text report generation

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
  - Reports RMSE, R2, and binary accuracy (when applicable)

- `src/NeuralNet.cpp` / `include/NeuralNet.h`
  - Dense feed-forward NN with ADAM/SGD options
  - Data-adaptive auto defaults (epochs, batch size, hidden width, dropout, patience)
  - Auto output/loss selection by inferred task type (classification vs regression)
  - Binary model serialization with integrity checks

### Dynamic Decision Layer
- `src/AutomationPipeline.cpp`
  - Auto target selection when `--target` is not provided
  - Strategy registry for target/feature/neural/bivariate policies
  - Auto task inference from target distribution (binary vs regression)
  - Adaptive sparse-feature gating with configurable threshold
  - Coherence-aware neural relevance scoring for bivariate selection
  - Auto decision log emitted into `neural_synthesis.txt`

### Numerical Core
- `src/MathUtils.cpp` / `include/MathUtils.h`
  - Cache-friendlier matrix multiplication (`A * B^T` row inner-products, OpenMP-aware)
  - Statistical significance utilities with guarded incomplete-beta fallback path
  - Contract-style API documentation for preconditions/postconditions/exceptions

### Plotting and Reporting
- `src/GnuplotEngine.cpp` / `include/GnuplotEngine.h`
  - Generates PNG plots in dedicated folders (`univariate`, `bivariate`, `overall`)
  - Safe id/path handling and script/value escaping
  - Automatically removes intermediate `.dat` / `.plt` files

- `src/ReportEngine.cpp` / `include/ReportEngine.h`
  - Plain-text report writer with section/table formatting

## Output

By default, each run generates:

- `univariate.txt` (univariate summary)
- `bivariate.txt` (all pair combinations + final significant table)
- `neural_synthesis.txt` (detailed neural lattice training trace + synthesis)
- `final_analysis.txt` (significant findings only, selected by neural decision engine)
- `seldon_report_assets/bivariate/` (PNG plots for selected significant bivariate results)

When supervised plotting is explicitly enabled, it also generates:

- `seldon_report_assets/univariate/` (univariate histograms/category distributions)
- `seldon_report_assets/overall/` (overall missingness/heatmap/loss trend plots)

## Config File (YAML/JSON-like key:value)

Example:

```yaml
target: sales
report: neural_synthesis.txt
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
verbose_analysis: true
neural_seed: 1337
gradient_clip_norm: 5.0
max_feature_missing_ratio: -1
target_strategy: auto
feature_strategy: auto
neural_strategy: auto
bivariate_strategy: auto
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
- `gradient_clip_norm` (>= 0)
- `max_feature_missing_ratio` (`-1` for auto, or fixed `[0,1]`)
- `target_strategy` (`auto`|`quality`|`max_variance`|`last_numeric`)
- `feature_strategy` (`auto`|`adaptive`|`aggressive`|`lenient`)
- `neural_strategy` (`auto`|`fast`|`balanced`|`expressive`)
- `bivariate_strategy` (`auto`|`balanced`|`corr_heavy`|`importance_heavy`)
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

## Robustness Notes

- Configuration values are validated early and fail fast with contextual errors.
- Report generation escapes HTML content to avoid malformed pages.
- Gnuplot scripts/ids are sanitized and plotting gracefully degrades if gnuplot is unavailable.
- Typed ingestion preserves non-numeric information for downstream analytics.
- Incomplete beta evaluation uses a continued-fraction primary path and a deterministic midpoint-integration fallback for difficult parameter regimes; this fallback is rarely invoked and can be slower for extreme parameters.
- Benchmark linear regression uses in-house Gaussian-elimination-based matrix routines suitable for baseline models; a dedicated linear algebra backend can be integrated later without changing public interfaces.
- Gnuplot invocation uses `std::system` in a local-tool context, with identifier sanitization and quoting/escaping to reduce command/script injection risk.
- Missing-value and outlier masks currently use `std::vector<bool>` for compactness; known proxy/performance quirks are acceptable at typical dataset scales.
- Datetime parsing uses strict deterministic ISO-like formats (`YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`) with leap-year/day validation.

## Current Scope

Seldon intentionally favors a strong automated foundation over advanced model complexity:

- Decision tree baseline currently uses a stump for speed and robustness.
- Config parser supports practical YAML/JSON-like key:value files, not full spec-complete parsing.
- `TypedDataset` is the production data path and orchestration targets full automation and dynamic model/feature selection.

## License

MIT (see `LICENSE`).
