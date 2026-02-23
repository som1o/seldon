# Architecture

Seldon is organized as a deterministic analytics pipeline around typed tabular data.

## Pipeline Overview

1. Ingest CSV into typed, row-aligned columns.
2. Preprocess data (missing values, outliers, scaling).
3. Run univariate statistics and benchmark models.
4. Train neural analysis (unless disabled by strategy).
5. Score/select significant bivariate pairs.
6. Generate reports and optional plot assets.

## Module Map

- `src/main.cpp`
  - CLI entrypoint and top-level invocation.

- `src/AutoConfig.cpp` + `include/AutoConfig.h`
  - Parses CLI and config-file values.
  - Merges values and runs centralized validation.
  - Supports lightweight YAML/JSON-like parsing with quote-aware handling.

- `src/TypedDataset.cpp` + `include/TypedDataset.h`
  - Streaming two-pass CSV loading.
  - Type inference for `numeric`, `categorical`, `datetime`.
  - Row-aligned typed storage and missing masks.
  - Configurable numeric separator policy for numeric parsing.

- `src/Preprocessor.cpp` + `include/Preprocessor.h`
  - Missing-value imputation by type.
  - Outlier detection on observed values before imputation.
  - Outlier action stage: `flag`, `remove`, `cap`.
  - Numeric scaling (`auto`, `zscore`, `minmax`, `none`).

- `src/Statistics.cpp` + `include/Statistics.h`
  - Core descriptive stats used across pipeline stages.

- `src/MathUtils.cpp` + `include/MathUtils.h`
  - Correlation/significance routines.
  - Matrix operations (transpose, multiply, inverse, QR).
  - Thread-local runtime numeric tuning.
  - Bounds-checked matrix accessors and inversion tolerance override hooks.

- `src/BenchmarkEngine.cpp` + `include/BenchmarkEngine.h`
  - K-fold benchmark baselines.
  - Produces RMSE/R² (and related diagnostics where applicable).

- `src/NeuralLayer.cpp` + `include/NeuralLayer.h`
  - Dense layer operations, activations, normalization, dropout, parameter updates.

- `src/NeuralNet.cpp` + `include/NeuralNet.h`
  - Feed-forward/backprop orchestration.
  - Optimizers (SGD/Adam/Lookahead), LR scheduling, clipping.
  - Feature-importance evaluation with adaptive runtime controls.

- `src/AutomationPipeline.cpp` + `include/AutomationPipeline.h`
  - End-to-end orchestration.
  - Target/feature/neural/bivariate strategy logic.
  - Bivariate selection and overall report assembly.
  - OpenMP thread-local collection for pair analysis.

- `src/GnuplotEngine.cpp` + `include/GnuplotEngine.h`
  - Plot generation backend.
  - Writes plot artifacts under report asset directories.

- `src/ReportEngine.cpp` + `include/ReportEngine.h`
  - Markdown report generation and section/table/image composition.

## Data Model

- `TypedColumn`
  - `name`
  - `type` (`NUMERIC`, `CATEGORICAL`, `DATETIME`)
  - `values` (`std::variant` over typed vectors)
  - `missing` (`std::vector<uint8_t>`)

- `TypedDataset`
  - Owns all columns and row count.
  - Maintains strict row alignment across transformations.

## Configuration Model

- `AutoConfig`
  - Runtime options (paths, plotting, strategies, neural controls).
  - Embedded `HeuristicTuningConfig` for thresholds and numeric tolerances.
  - Single `validate()` method enforces invariants after parsing.

## Key Design Decisions

- Typed, columnar storage with explicit missing masks.
- Deterministic, fail-fast configuration validation.
- Outlier detection isolated from imputed synthetic values.
- Correlation heatmap capped to avoid unbounded $O(n^2)$ behavior on wide data.
- Thread-local accumulation in OpenMP regions to avoid critical-section bottlenecks.
- Numeric runtime controls exposed through config and applied centrally.

## Parallelism

When built with OpenMP enabled:

- Matrix and layer loops use OpenMP where beneficial.
- Bivariate pair generation uses thread-local vectors and concatenation.
- Non-OpenMP builds fall back to serial execution.

## Numerical Robustness

- Welford online variance in quality scoring paths.
- Thread-local numeric epsilon and beta fallback tuning.
- Robust rank/pivot tolerance heuristics with matrix override support.
- Significance and correlation paths share centralized math utilities.

## Reporting Flow

- Core markdown reports are always produced.
- Plot sections are included when plotting is enabled and backend tools are available.
- Final analysis report includes selected significant findings only.

## Extension Points

- Add new benchmark models in `BenchmarkEngine`.
- Add new feature-selection policies in `AutomationPipeline`.
- Swap plotting backend while preserving report contract.
- Extend typed datetime feature extraction in `TypedDataset` and preprocessing.

## Related Docs

- [README.md](../README.md)
- [docs/USAGE.md](USAGE.md)
