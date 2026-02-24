# Architecture

This document describes the runtime architecture of Seldon in operational detail.

It is intended for contributors and operators who need to understand:

- runtime boundaries,
- module responsibilities,
- data contracts,
- resource behavior,
- and output semantics.

---

## 1) Architectural Intent

Seldon is a staged analytical pipeline.

The architecture is designed around these goals:

- deterministic stage ordering,
- explicit typed data representation,
- configurable behavior without code edits,
- robust handling of real-world malformed tabular data,
- bounded computational behavior through guardrails,
- report-first output suitable for batch workflows.

---

## 2) High-Level Runtime Flow

The orchestrated flow is:

1. Parse runtime configuration.
2. Validate config invariants.
3. Resolve input source and conversion path.
4. Load typed dataset.
5. Apply optional explicit type overrides.
6. Preprocess data.
7. Select target and features.
8. Run baseline benchmarks.
9. Run neural analysis.
10. Compute coherent importance.
11. Perform bivariate analysis and selection.
12. Generate report artifacts.
13. Optionally generate HTML artifacts.

The controller for this flow lives in:

- `src/AutomationPipeline.cpp`

---

## 3) Module Layout

### 3.1 Orchestration and Configuration

- `include/AutomationPipeline.h`
- `src/AutomationPipeline.cpp`
- `include/AutoConfig.h`
- `src/AutoConfig.cpp`

Responsibilities:

- CLI parsing,
- config file merging,
- config validation,
- strategy selection,
- end-to-end sequencing,
- final artifact coordination.

### 3.2 Data Representation and Ingestion

- `include/TypedDataset.h`
- `src/TypedDataset.cpp`
- `include/CSVUtils.h`
- `src/CSVUtils.cpp`

Responsibilities:

- source conversion handling,
- CSV tokenization,
- row width reconciliation,
- type inference,
- typed value storage,
- missingness masks,
- explicit type overrides.

### 3.3 Preprocessing

- `include/Preprocessor.h`
- `src/Preprocessor.cpp`

Responsibilities:

- missing value accounting,
- imputation,
- outlier detection,
- outlier action application,
- scaling,
- temporal feature extraction,
- controlled numeric feature expansion.

### 3.4 Statistics and Math

- `include/Statistics.h`
- `src/Statistics.cpp`
- `include/MathUtils.h`
- `src/MathUtils.cpp`

Responsibilities:

- descriptive statistics,
- correlation and significance,
- numeric summarization,
- matrix algebra,
- inversion and regression utilities,
- runtime numeric tolerance controls.

### 3.5 Modeling and Relevance

- `include/BenchmarkEngine.h`
- `src/BenchmarkEngine.cpp`
- `include/NeuralLayer.h`
- `src/NeuralLayer.cpp`
- `include/NeuralNet.h`
- `src/NeuralNet.cpp`

Responsibilities:

- baseline k-fold benchmarks,
- neural training,
- explainability,
- uncertainty estimation,
- safety limits for topology and parameters.

### 3.6 Reporting and Plotting

- `include/ReportEngine.h`
- `src/ReportEngine.cpp`
- `include/GnuplotEngine.h`
- `src/GnuplotEngine.cpp`

Responsibilities:

- Markdown section composition,
- table and image insertion,
- plot script generation and rendering,
- output artifact placement.

---

## 4) Core Data Contracts

### 4.1 `TypedColumn`

`TypedColumn` contains:

- `name`,
- `type` (`NUMERIC|CATEGORICAL|DATETIME`),
- `values` (variant storage),
- `missing` mask (`std::vector<uint8_t>`).

Design implications:

- row alignment preserved for all columns,
- type-safe data access via variant,
- explicit missingness decoupled from value payload.

### 4.2 `TypedDataset`

`TypedDataset` contains:

- filename,
- delimiter,
- locale hints,
- optional column type override map,
- row count,
- typed columns.

Key API behaviors:

- load infers and materializes schema,
- lookup by name is supported,
- row filtering preserves alignment across columns.

---

## 5) Ingestion Design Details

### 5.1 Source Resolution

Input path is resolved by extension.

If conversion is required,
the file is converted to a temporary CSV stream target.

This allows the rest of ingestion logic to stay format-agnostic.

### 5.2 Row Width Reconciliation

Real-world CSV data often contains irregular row width.

Seldon applies reconciliation logic to:

- resize underfull rows,
- merge overfull token pairs heuristically,
- handle date-shift anomalies,
- skip obvious control/metadata rows.

### 5.3 Multi-Pass Inference

The loader performs an inference pass and follow-up passes to:

- infer candidate type,
- populate typed vectors,
- fallback datetime columns to categorical when parsing quality is poor,
- rescue likely datetime semantics when strongly detected.

Explicit type overrides short-circuit fallback behaviors for forced columns.

---

## 6) Type Inference and Override Interaction

Default behavior:

- infer from observed token distributions and parseability.

Override behavior:

- explicit `type.<column>` or `--type` maps take precedence.

Rationale:

- inference handles common cases,
- overrides handle domain-specific ambiguities,
- combined model preserves both convenience and control.

---

## 7) Preprocessing Pipeline Details

Preprocessing executes in this order:

1. Date-derived feature synthesis.
2. Auto numeric feature expansion (bounded).
3. Outlier flag detection on observed values.
4. Missing count bookkeeping and imputation.
5. Outlier action application.
6. Scaling.

### 7.1 Outlier Detection Scope

Outlier flags are computed prior to imputation,
so synthetic fill values do not influence anomaly detection.

### 7.2 Memory Guard in Outlier Reporting

Row-level outlier flags can be large.

Storage is now controlled by:

- `storeOutlierFlagsInReport`.

### 7.3 Feature Expansion Guard

Feature expansion can grow quickly.

Seldon enforces:

- `featureEngineeringMaxGeneratedColumns`.

This cap is applied while generating polynomial,
log,
and interaction features.

---

## 8) Feature Selection and Strategy Layer

Feature selection behavior is strategy-driven.

Inputs include:

- missingness,
- variance,
- leakage thresholds,
- strategy mode (`adaptive|aggressive|lenient|auto`).

Outcome:

- retained feature set,
- dropped-feature diagnostics,
- threshold metadata used in reports.

---

## 9) Neural Input Encoding Architecture

Neural input matrix is built from:

- selected numeric features,
- one-hot encoded categorical columns,
- optional `other` category bucket.

Controls:

- `neuralMaxOneHotPerColumn`.

Memory/perf behavior:

- precomputed encoding plan,
- row-level reserve by expected encoded width,
- reduced dynamic reallocation pressure.

---

## 10) Benchmark Engine Design

Benchmark engine compares baseline models using fold-based evaluation.

Key outputs:

- RMSE,
- R²,
- optional accuracy for binary-like targets,
- actual/predicted vectors for report surfacing.

Parallel behavior:

- model sections can run concurrently with OpenMP.

Concurrency safety:

- explicit OpenMP sharing clauses are used in parallel sections.

---

## 11) Neural Engine Design

### 11.1 Layer Primitive (`DenseLayer`)

Responsibilities:

- forward projection,
- activation application,
- dropout masking,
- optional normalization transforms,
- gradient storage,
- parameter updates.

### 11.2 Network Runtime (`NeuralNet`)

Responsibilities:

- training loop,
- batch stepping,
- validation path,
- LR scheduling,
- early stopping,
- lookahead/adam/sgd optimization,
- explainability scoring,
- uncertainty estimation.

### 11.3 Topology Safety

Two safety layers exist:

- config-level limits in orchestration,
- hard constructor-level limits in `NeuralNet`.

Limits constrain:

- total node count,
- trainable parameter count.

This blocks over-allocation risks from malicious or accidental config values.

---

## 12) Explainability and Importance Fusion

Modes:

- permutation,
- integrated gradients,
- hybrid weighted blend.

Hybrid mode uses configurable weights:

- permutation weight,
- integrated gradients weight.

Weights are validated and normalized during scoring.

---

## 13) Coherent Importance Layer

Seldon computes coherent importance by blending:

- neural feature importance,
- target-feature correlation signal.

Blend weighting adjusts using:

- dataset size,
- overfitting proxy,
- benchmark-vs-neural penalty heuristics.

This stabilizes feature prioritization under noisy conditions.

---

## 14) Bivariate Analysis Architecture

Bivariate pair generation evaluates numeric pair statistics:

- Pearson,
- Spearman,
- Kendall,
- significance,
- regression descriptors,
- neural relevance score.

Selection process:

- significance filter,
- neural-score quantile cutoff,
- capped keep count.

Parallel implementation:

- per-thread pair buffers,
- post-join vector merge,
- explicit OpenMP sharing declarations.

---

## 15) Plotting and Visual Artifact Layer

Plot generation is delegated to `GnuplotEngine`.

Plot modes include:

- univariate,
- bivariate significant,
- overall diagnostics.

Artifacts are written to deterministic subfolders under:

- `seldon_report_assets`.

---

## 16) Reporting Architecture

`ReportEngine` acts as a markdown builder.

Report composition pattern:

- add title,
- add paragraphs,
- add tables,
- add image references,
- save markdown.

Primary report outputs:

- univariate report,
- bivariate report,
- neural synthesis report,
- final analysis report.

Optional HTML conversion is delegated to `pandoc` when enabled.

---

## 17) Logging and Terminal UX

Runtime messaging uses:

- stage progress updates,
- verbose analytics logs,
- optional spinner updates.

Current behavior avoids mixed-line collisions by enabling spinner only in non-verbose TTY runs.

---

## 18) OpenMP Boundaries

OpenMP is conditionally compiled.

Key parallel zones include:

- benchmark model sections,
- pairwise bivariate evaluation,
- selected neural importance loops,
- matrix multiplication paths.

Concurrency hygiene:

- thread-local accumulation where needed,
- explicit `default(none)` and `shared(...)` in sensitive regions.

---

## 19) Numeric Stability Controls

`MathUtils` centralizes runtime numeric controls:

- significance alpha,
- numeric epsilon,
- beta function fallback intervals,
- fallback tolerance.

These controls are used across significance,
matrix inversion,
and related numerical routines.

---

## 20) Configuration Validation Layer

`AutoConfig::validate` and `HeuristicTuningConfig::validate` enforce:

- option domain correctness,
- range constraints,
- cross-field consistency,
- and now safety limits for neural topology parameters.

Invalid settings fail fast before heavy execution begins.

---

## 21) Resource Behavior Summary

Main memory consumers are:

- loaded dataset columns,
- engineered feature expansions,
- encoded neural matrix,
- report intermediate vectors.

Implemented containment controls include:

- generated-feature cap,
- one-hot-per-column cap,
- optional outlier-flag retention,
- neural topology/parameter limits,
- fast-mode sample caps.

---

## 22) Failure Domains

Primary failure categories:

- configuration errors,
- dataset parse errors,
- external converter absence,
- numeric edge cases,
- unsupported settings combinations.

Seldon surfaces descriptive exceptions with context.

---

## 23) Artifact Contract

When successful, Seldon emits a consistent artifact set.

Mandatory markdown artifacts:

- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`

Asset directory:

- `seldon_report_assets/`

Optional exports:

- preprocessed CSV/Parquet,
- optional HTML report counterparts.

---

## 24) Extensibility Surface

Most runtime behavior extension points are config-driven:

- parser hints,
- per-column overrides,
- per-column imputation,
- strategy knobs,
- heuristic thresholds,
- safety boundaries.

This architecture minimizes source-level changes for common behavior tuning.

---

## 25) Practical Mental Model

Think of Seldon as five stacked layers:

1. **Ingestion layer**
2. **Transformation layer**
3. **Analysis layer**
4. **Selection layer**
5. **Reporting layer**

Each layer has strict data contracts and bounded responsibilities.

This separation helps maintain predictable behavior as data scale and schema complexity increase.

---

## 26) File-Level Quick Index

### 26.1 Entry and Pipeline

- `src/main.cpp`
- `src/AutomationPipeline.cpp`

### 26.2 Config

- `include/AutoConfig.h`
- `src/AutoConfig.cpp`

### 26.3 Data and Preprocessing

- `include/TypedDataset.h`
- `src/TypedDataset.cpp`
- `include/Preprocessor.h`
- `src/Preprocessor.cpp`

### 26.4 Stats and Math

- `include/Statistics.h`
- `src/Statistics.cpp`
- `include/MathUtils.h`
- `src/MathUtils.cpp`

### 26.5 Modeling

- `include/BenchmarkEngine.h`
- `src/BenchmarkEngine.cpp`
- `include/NeuralLayer.h`
- `src/NeuralLayer.cpp`
- `include/NeuralNet.h`
- `src/NeuralNet.cpp`

### 26.6 Report and Plot

- `include/ReportEngine.h`
- `src/ReportEngine.cpp`
- `include/GnuplotEngine.h`
- `src/GnuplotEngine.cpp`

---

## 27) Cross-Document Links

- [README](../README.md)
- [USAGE](USAGE.md)
- [ENABLE_EXCEL_IMPORT](ENABLE_EXCEL_IMPORT.md)
