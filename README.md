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
cmake ..
cmake --build . -j
```

Legacy compatibility build (optional):

```bash
cmake -DSELDON_ENABLE_LEGACY=ON ..
cmake --build . -j
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
  - Reports RMSE, R2, and binary accuracy (when applicable)

- `src/NeuralNet.cpp` / `include/NeuralNet.h`
  - Dense feed-forward NN with ADAM/SGD options
  - Early stopping and LR decay defaults
  - Binary model serialization with integrity checks

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

- `univaraite.txt` (univariate summary)
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
plots: bivariate
exclude: id,notes
impute.sales: median
impute.region: mode
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
- `plots` (comma-separated profile: `none`, `bivariate`, `univariate`, `overall`, `all`)
- `exclude` (comma-separated)
- `impute.<columnName>`

## Robustness Notes

- Configuration values are validated early and fail fast with contextual errors.
- Report generation escapes HTML content to avoid malformed pages.
- Gnuplot scripts/ids are sanitized and plotting gracefully degrades if gnuplot is unavailable.
- Typed ingestion preserves non-numeric information for downstream analytics.
- Incomplete beta evaluation uses a continued-fraction primary path and a deterministic midpoint-integration fallback for difficult parameter regimes; this fallback is rarely invoked and can be slower for extreme parameters.
- Benchmark linear regression uses in-house Gaussian-elimination-based matrix routines suitable for baseline models; a dedicated linear algebra backend can be integrated later without changing public interfaces.
- Gnuplot invocation uses `std::system` in a local-tool context, with identifier sanitization and quoting/escaping to reduce command/script injection risk.
- Missing-value and outlier masks currently use `std::vector<bool>` for compactness; known proxy/performance quirks are acceptable at typical dataset scales.
- Datetime parsing currently uses `std::mktime`; it mutates the provided `std::tm` and depends on locale/timezone globals, but this path is not used concurrently in the current pipeline.

## Current Scope

Seldon intentionally favors a strong automated foundation over advanced model complexity:

- Decision tree baseline currently uses a stump for speed and robustness.
- Config parser supports practical YAML/JSON-like key:value files, not full spec-complete parsing.
- `TypedDataset` is the primary path; legacy `Dataset`/`LogicEngine`/`StatsEngine`/`TerminalUI` modules remain for compatibility, are excluded from default builds, and are candidates for future retirement.

## License

MIT (see `LICENSE`).
