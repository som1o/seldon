# Architecture

## 1. Architectural Summary

Seldon is organized as a staged, deterministic analytics system for tabular data. The architecture centers on a typed columnar data model and a pipeline controller that coordinates ingestion, preprocessing, statistical analysis, model evaluation, neural relevance analysis, and report materialization.

Execution is configured through a unified runtime configuration model (`AutoConfig`) populated from command-line arguments and optional configuration files.

## 2. Processing Pipeline

The runtime sequence is:

1. Parse and validate runtime configuration.
2. Load CSV data into a typed, row-aligned dataset representation.
3. Apply preprocessing transforms (missingness, outliers, scaling, derived fields).
4. Compute descriptive and inferential statistics.
5. Evaluate baseline predictive models through fold-based benchmarking.
6. Execute neural training and compute feature relevance metrics (when enabled by strategy).
7. Select significant bivariate relations through statistical and relevance criteria.
8. Generate Markdown reports and optional plot/HTML artifacts.

## 3. Core Modules

### 3.1 Runtime and Orchestration

- `src/main.cpp`: process entry point and top-level invocation.
- `src/AutomationPipeline.cpp` (`include/AutomationPipeline.h`): stage orchestration, strategy dispatch, bivariate selection, and final report assembly.
- `src/AutoConfig.cpp` (`include/AutoConfig.h`): argument/config parsing, merge semantics, and invariant validation.

### 3.2 Data Ingestion and Transformation

- `src/TypedDataset.cpp` (`include/TypedDataset.h`): two-pass CSV loading, type inference, typed column storage, and missing-value masks.
- `src/Preprocessor.cpp` (`include/Preprocessor.h`): imputation, outlier handling, scaling, and preprocessing policy execution.

### 3.3 Statistics and Numerical Methods

- `src/Statistics.cpp` (`include/Statistics.h`): descriptive and dataset-level statistical routines.
- `src/MathUtils.cpp` (`include/MathUtils.h`): matrix operations, significance/correlation utilities, and numerical tolerance controls.

### 3.4 Predictive and Relevance Engines

- `src/BenchmarkEngine.cpp` (`include/BenchmarkEngine.h`): fold-based baseline model benchmarking (e.g., RMSE, $R^2$).
- `src/NeuralLayer.cpp` (`include/NeuralLayer.h`): dense-layer primitives, activations, normalization, dropout, and updates.
- `src/NeuralNet.cpp` (`include/NeuralNet.h`): network training loop, optimizer logic, scheduling, clipping, and feature-importance estimation.

### 3.5 Reporting and Visualization

- `src/ReportEngine.cpp` (`include/ReportEngine.h`): Markdown report composition.
- `src/GnuplotEngine.cpp` (`include/GnuplotEngine.h`): plot script generation and rendering through `gnuplot`.

## 4. Data Model

### 4.1 TypedColumn

Each column stores:

- a name,
- a type (`NUMERIC`, `CATEGORICAL`, `DATETIME`),
- a typed value container, and
- a missingness mask.

### 4.2 TypedDataset

The dataset object owns all columns and the row cardinality. Row alignment is maintained across all transforms to preserve correspondence among features and target values.

## 5. Configuration Model

`AutoConfig` contains runtime settings for:

- data source and parsing,
- preprocessing policy,
- strategy selection,
- neural and benchmark controls,
- plotting/output behavior, and
- numeric robustness thresholds.

Validation is centralized and executed after parse/merge to reject invalid configurations before pipeline execution.

## 6. Parallel Execution

When compiled with OpenMP support, selected computational loops (including matrix operations and pairwise analysis paths) execute in parallel. Parallel regions use thread-local accumulation where required to reduce contention. Without OpenMP, the same logic executes serially.

## 7. Numerical Behavior

The architecture applies centralized numeric controls (epsilon and tolerance parameters) to reduce instability in matrix and significance calculations. Outlier detection is applied to observed numeric values prior to imputation in order to isolate raw-signal anomaly detection from synthetic fill values.

## 8. Output Contract

The system emits four primary Markdown documents (`univariate.md`, `bivariate.md`, `neural_synthesis.md`, `final_analysis.md`). Plot assets and HTML reports are conditionally generated based on runtime configuration and external tool availability.

## 9. Cross-Reference

- [Project Overview](../README.md)
- [Usage Reference](USAGE.md)
