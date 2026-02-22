# Seldon

Seldon is a C++ automated analytics engine that performs end-to-end data processing with minimal user input:

- Typed CSV ingestion (numeric, categorical, datetime)
- Automated preprocessing (missing values, outliers, scaling)
- Automated EDA plotting via gnuplot
- Baseline model benchmarking (linear, ridge, tree-stump)
- Neural-network training with sensible defaults
- Unified HTML report generation

## Build

```bash
mkdir -p build
cd build
cmake ..
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
```

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
  - Generates histogram, bar, scatter, line, heatmap
  - Safe id/path handling and script/value escaping

- `src/ReportEngine.cpp` / `include/ReportEngine.h`
  - Unified HTML report templating
  - HTML escaping for robust rendering

## Output

By default, each run generates:

- `seldon_report.html`
- `seldon_report_assets/` (all generated plot files and intermediate gnuplot files)
- `seldon_model.seldon` (trained neural model)

## Config File (YAML/JSON-like key:value)

Example:

```yaml
target: sales
report: seldon_report.html
assets_dir: seldon_report_assets
delimiter: ,
outlier_method: iqr
outlier_action: flag
scaling: auto
kfold: 5
plot_format: png
plot_width: 1280
plot_height: 720
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

## License

MIT (see `LICENSE`).
