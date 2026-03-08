# Seldon

Seldon is a self-contained, end-to-end automated data analytics and machine learning pipeline
written in C++. It ingests a CSV/Excel dataset, runs a full technical analysis — covering type
inference, preprocessing, statistical profiling, deterministic modeling, neural network
training, causal discovery, and artifact generation — and emits structured Markdown reports,
plots, and optional HTML output without requiring any external runtime language environment.

Seldon is built for engineers and analysts who want rigorous, reproducible analysis as a local
batch operation or as a step in a CI pipeline, with no Python interpreter, no Jupyter server,
and no managed cloud dependency on the critical path.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Building Seldon](#building-seldon)
- [Running Seldon](#running-seldon)
- [Output Artifacts](#output-artifacts)
- [CLI Reference](#cli-reference)
- [Configuration File Reference](#configuration-file-reference)
- [Runtime Modes](#runtime-modes)
- [Statistical Scope and Guardrails](#statistical-scope-and-guardrails)
- [Troubleshooting](#troubleshooting)

---

## Introduction

Most analytical toolchains require stitching together a preprocessing library, a feature
selection routine, a modelling framework, a plotting backend, and a report renderer — in a
scripting language that carries its own runtime versioning and dependency management burden.
Seldon collapses that entire stack into a single compiled binary.

Feed it a CSV file. Seldon infers column types, handles missing data and outliers, computes a
full univariate and bivariate statistical profile, selects features deterministically, trains a
dense neural network with explainability hooks, runs observational causal discovery, generates
plots via gnuplot, assembles structured Markdown reports, and optionally converts everything to
HTML via pandoc.

The design priorities are:

- **Determinism first.** Statistical and deterministic passes precede and constrain neural
  analysis. Core computations are reproducible across runs given the same input and
  configuration.
- **Graceful degradation.** Optional build backends — OpenMP, OpenCL, CUDA, native
  Parquet — are detected at build time and absent backends are silently omitted without
  changing the binary's functional contract.
- **Low operational overhead.** A single statically-linked binary, no runtime interpreter,
  no model registry server, no dependency injection container. The output directory can be
  zipped and shared as a self-contained artefact.
- **Configurable depth.** The full pipeline runs by default. Fast mode and low-memory mode
  cap expensive pair analysis and model complexity for quick iteration.

---

## Features

### Data Loading and Type Inference

- Pure C++ CSV parser with a single-pass state machine and configurable field, record, and
  column count limits.
- Automatic column type inference: numeric (floating point or integer), categorical (string),
  and datetime. Inference applies a ranked parse-attempt strategy — numeric first, datetime
  second, categorical as fallback — with confidence thresholds and early bailout.
- Datetime parsing covers ISO 8601, DMY, MDY, and locale-aware formats with multiple
  separator variants.
- Numeric separator policy selection for US (`1,000.0`) and European (`1.000,0`) formats.
- BOM handling and memory-mapped file reads for large inputs; bounded streaming chunk mode
  for minimal heap allocation.
- Schema override support: force a column type or exclude a column from analysis without
  changing the source file.

### Preprocessing

- Per-column missing value detection and configurable imputation strategies: mean, median,
  mode, and KNN imputation. KNN builds a k-d tree over valid numeric rows with O(n log n)
  construction and O(k log n) query time; missing values are interpolated from the k nearest
  neighbors in feature space.
- Outlier detection via strict Local Outlier Factor (LOF) scoring with configurable
  contamination thresholds or IQR-based rules. The `lof_fallback_modified_zscore` key is
  retained as a legacy alias and routes to the same strict LOF code path.
- Scaling options: standard (z-score), min-max, and robust (median/IQR-based) normalization.
- A `PreprocessReport` is generated per run recording: missing value counts per column,
  outlier row counts, imputation choices, and scaling parameters — making the preprocessing
  pass fully auditable.

### Univariate Statistical Profiling

- Full column-level profile for every numeric variable: mean, median, variance, standard
  deviation, skewness, kurtosis, IQR, configurable percentile array, min, max, and missing
  rate.
- Variance computed via Welford's online algorithm for numerical stability.
- ADF-style drift detection using linear regression on temporal proxies.
- Entropy estimation via quantile binning (8–16 bins, configurable).
- Distribution shape and modality heuristics.
- Per-column plots: histograms (Sturges rule bin count), box plots, KDE approximations.

### Bivariate Analytics and Pair Diagnostics

- Pairwise correlation matrix with three correlation measures: Pearson (O(n)), Spearman
  rank correlation (O(n log n) with tie-handling via `nth_element`), and Kendall Tau.
- Mutual information estimation via histogram binning.
- Interaction effect scoring and drift-based pair diagnostics.
- Configurable pair cap and sample limits for large feature spaces. In fast mode, pair
  analysis is capped automatically to stay within iteration time budgets.
- Bivariate scatter plots and heatmaps generated to the assets directory.

### Feature Selection and Deterministic Heuristics

- Row-to-feature ratio gating: features are excluded if the dataset is too narrow to support
  reliable estimation.
- Leakage detection: columns whose names match known metadata, administrative, or temporal
  indicator patterns are flagged and optionally excluded.
- Missing-rate threshold filtering per column.
- Lasso-based coordinate descent feature selection: 80 iterations of soft-thresholding with
  feature scaling, producing a ranked importance list for downstream neural analysis.
- Feature role tagging by name-pattern matching (regex-based, case-insensitive): datetime
  proxies, identifier columns, audit fields.
- Feature engineering: polynomial terms, log transforms, and pairwise interaction columns
  generated on demand.

### Deterministic Baseline Modeling

- A benchmark suite of classical baseline models runs before the neural training stage,
  establishing reproducible performance references: linear regression, ridge regression, and
  tree-based heuristics.
- K-fold cross-validation with configurable fold count; metrics per fold and aggregate:
  RMSE, R², accuracy.
- Feature importance scores computed from baseline models and cached for neural
  initialization hints.
- `BenchmarkResult` structs capture per-model, per-fold metrics and are embedded in the
  final report.

### Neural Network Training

- Dense feed-forward network with configurable topology (layer count and width per layer).
- Activation functions: Sigmoid, ReLU, Tanh, Linear, GELU (tanh polynomial approximation).
- Optimizers: SGD, Adam (with bias correction), and Lookahead wrapper.
- Regularization: dropout (Bernoulli gate with splitmix64 PRNG), L1/L2 penalties, gradient
  clipping with adaptive norm tracking.
- Normalization: batch normalization (running statistics) and layer normalization
  (per-sample reduce and scale), both configurable per layer.
- Exponential moving average (EMA) weight tracking for smoothed final inference.
- Gradient accumulation for larger effective batch sizes on memory-constrained runs.
- Over 40 tunable hyperparameters via the `Hyperparameters` struct: learning rate, decay
  schedule, warmup steps, dropout rates, batch and layer norm epsilon, EMA decay, gradient
  clip threshold, and more.
- Optional `SELDON_NEURAL_FLOAT32` compile flag to reduce network memory footprint and
  increase throughput on float32-capable hardware at the cost of precision.

### Explainability

- Permutation-based feature importance scores computed post-training for each input feature.
- Partial dependence estimates for top features.
- Residual analysis with outlier contextualization: residual narrative sections in the report
  identify anomalous predictions with their contributing feature values.
- Model uncertainty estimates from dropout inference at test time.
- A structured readiness matrix is included in generated reports, labeling each analytical
  subsystem as `stable`, `experimental`, or `best-effort`.

### Causal Discovery

- Observational causal graph inference using an ensemble of algorithms: PC algorithm with
  Meek orientation rules, Fast Causal Inference (FCI), Greedy Equivalence Search (GES), and
  a LiNGAM heuristic (iterative Lasso-based).
- Conditional independence (CI) testing via partial correlation (residual regression + Fisher
  z-transform). An HSIC kernel test (O(n²) kernel matrix) is applied near decision boundaries
  for non-linear dependence detection.
- Bootstrap validation of edge presence and directionality.
- Temporal proxy validation when datetime columns are present.
- Algorithmic consensus across methods gives each directed edge a support score.
- Causal graph output is embedded in the final analysis report. Directionality is explicitly
  framed as a hypothesis under observational assumptions, not an intervention-level proof.

### Report Generation

- Five structured Markdown report files per run: `univariate.md`, `bivariate.md`,
  `report.md`, `final_analysis.md`, and a configurable neural synthesis report
  (default: `neural_synthesis.md`).
- Plot assets are organized under `seldon_report_assets/` with `univariate/`, `bivariate/`,
  and `overall/` subdirectories. All image links in reports are normalized relative to the
  report file location for portability.
- Optional HTML conversion of all Markdown outputs via pandoc. HTML export reports explicit
  conversion success and failure counts per file; no silent best-effort behavior.
- Optional Parquet export of preprocessed data using a native C++ Arrow/Parquet backend
  (no Python bridge). Binaries built without Arrow emit a clear runtime warning and retain
  CSV export.
- Benchmark mode: per-stage wall-clock timings and pair-analysis throughput metrics are
  embedded in the report when `--benchmark-mode true` is passed.

### Acceleration Backends

- **OpenMP**: parallel neuron forward pass, weight update loops, and SIMD vectorization
  pragmas on hot preprocessing paths. Enabled by default.
- **OpenCL**: optional GPU-accelerated paths. Detected at build time; builds without OpenCL
  behave identically to OpenMP-only builds.
- **CUDA**: optional secondary `seldon_cuda` binary target when CUDA toolkit is available.
- All acceleration backends are optional and non-blocking: their absence never changes the
  analytical output.

---

## Architecture

### Pipeline Stages

The entry point is `AutomationPipeline::run()`. Execution is divided into ten sequential
stages:

```
Stage 1   Resolve runtime config and output paths
Stage 2   Clean up output and artifact directories
Stage 3   Load typed dataset (CSV parse, type inference, schema validation)
Stage 4   Target resolution + preprocessing (imputation, outliers, scaling)
Stage 5   Univariate profiling and column-level statistics
Stage 6   Feature selection and deterministic heuristic guards
Stage 7   Benchmark suite + neural network training and evaluation
Stage 8   Bivariate pair analytics and advanced diagnostics
Stage 9   Report assembly (Markdown composition and HTML export)
Stage 10  Artifact save, completion summary, benchmark timings
```

Stages 1–6 are deterministic and form a strict dependency chain. Stages 7–8 are analytically
independent at the algorithmic level but are sequenced to feed neural explainability scores
into the bivariate report. Stage 9 consumes all prior outputs.

### Source Layout

```
src/
  AutomationPipeline.cpp       — Top-level orchestrator, includes pipeline parts
  pipeline_parts/
    PipelineRuntime.cpp        — End-to-end execution flow and report composition
    PipelineUnivariate.cpp     — Univariate scaffolding, plot helpers, I/O save helpers
    PipelineModeling.cpp       — Neural training strategies, explainability, scoring policy
    PipelineBivariate.cpp      — Pair analytics, interaction/drift/context diagnostics
    PipelineReporting.cpp      — Residual narrative and outlier contextualization
  TypedDataset.cpp             — Columnar typed storage, CSV loading, type inference
  Preprocessor.cpp             — Missing/outlier handling, scaling, k-d tree KNN imputation
  Statistics.cpp               — Descriptive and inferential statistics
  MathUtils.cpp                — Numeric utilities, linear algebra primitives
  NeuralNet.cpp                — Network topology, forward/backward pass, optimizers
  NeuralLayer.cpp              — Dense layer implementation, activations, SIMD loops
  BenchmarkEngine.cpp          — Baseline model family, K-fold CV, metrics
  DeterministicHeuristics.cpp  — Feature filtering, Lasso selection, role tagging
  CausalDiscovery.cpp          — CI testing, PC/FCI/GES/LiNGAM, bootstrap validation
  GnuplotEngine.cpp            — Script generation, subplot management, process spawning
  PlotHeuristics.cpp           — Bin count rules, downsampling, adaptive axis inference
  ReportEngine.cpp             — Markdown assembly, HTML export via pandoc
  StatsUtils.cpp               — Shared statistical helpers
  CSVUtils.cpp                 — Low-level CSV tokenizer and field parser
  AutoConfig.cpp               — Configuration struct loading and CLI flag parsing
```

### Core Data Structures

**`TypedDataset`** is the central data container throughout the pipeline. It holds typed
columns as a `std::variant` over `std::vector<double>`, `std::vector<std::string>`, and
`std::vector<int64_t>`, with a parallel boolean missing mask per column. Columns carry their
inferred type (`NUMERIC`, `CATEGORICAL`, `DATETIME`), name, and per-value missing flags.

**`TypedColumn`**:
```cpp
struct TypedColumn {
    std::string name;
    ColumnType  type;
    std::variant<std::vector<double>,
                 std::vector<std::string>,
                 std::vector<int64_t>> values;
    std::vector<bool> missing;  // true = missing
};
```

**`NeuralNet`** owns a sequence of `NeuralLayer` objects, each storing weight matrices,
bias vectors, activation type, normalization state, and gradient accumulators. The network
supports forward and backward passes, EMA weight snapshots, and dropout masks.

**`AutoConfig`** carries over 100 configuration parameters parsed from a JSON file and/or
overridden by CLI flags. It is the single source of truth for all tunable behavior and is
passed by const reference to every pipeline stage.

**`HeuristicTuningConfig`** carries 37 threshold parameters for statistical tests, feature
filtering rules, and selection cutoffs.

### Output Path Resolution

Path resolution is centralized in `PipelineRuntime.cpp`:

- If `--output-dir` is not set, the output root is derived as `<dataset_stem>_seldon_outputs`
  beside the input file.
- `--assets-dir`: if empty or default, resolved to `<output_dir>/seldon_report_assets`. If
  relative, resolved under `output_dir`. If absolute, used as-is.
- `--report`: if empty or default, resolved to `<output_dir>/neural_synthesis.md`. If
  relative, resolved under `output_dir`. If absolute, used as-is.
- All image and document cross-links in Markdown output are normalized to paths relative to
  each report file's own location, making the output directory portable across filesystem
  roots without broken links.

---

## Prerequisites

**Required:**

- Linux or macOS
- C++17 compiler: GCC 9+ or Clang 9+
- CMake 3.10 or newer

**Optional (detected at build time):**

| Dependency       | Purpose                                          |
|------------------|--------------------------------------------------|
| OpenMP           | Parallel neural layers, SIMD preprocessing       |
| OpenCL ICD + dev | GPU-accelerated compute paths                    |
| CUDA toolkit     | Secondary `seldon_cuda` binary target            |
| Arrow/Parquet C++| Native Parquet export without Python bridge      |
| `gnuplot`        | Plot generation (must be on PATH at runtime)     |
| `pandoc`         | HTML report export (must be on PATH at runtime)  |

---

## Building Seldon

### Standard Release Build

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSELDON_ENABLE_OPENMP=ON \
  -DSELDON_ENABLE_OPENCL=OFF
cmake --build build -j"$(nproc)"
```

### With OpenCL Acceleration

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSELDON_ENABLE_OPENMP=ON \
  -DSELDON_ENABLE_OPENCL=ON
cmake --build build -j"$(nproc)"
```

When OpenCL is successfully found, CMake prints:

```
Seldon: OpenCL enabled
```

If no ICD or headers are found, the build completes normally without the OpenCL path.
Analytical output is identical.

### With CUDA

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSELDON_ENABLE_CUDA=ON
cmake --build build -j"$(nproc)"
```

When the CUDA toolkit is present, this produces both `seldon` and `seldon_cuda` under
`build/`.

### With Native Parquet Export

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSELDON_ENABLE_NATIVE_PARQUET=ON
cmake --build build -j"$(nproc)"
```

Requires Arrow and Parquet C++ libraries installed and discoverable by CMake. Without them
the build proceeds without native Parquet support; the binary will emit a diagnostic warning
at runtime if Parquet export is requested.

### Float32 Neural Network

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSELDON_NEURAL_FLOAT32=ON
cmake --build build -j"$(nproc)"
```

Configures the neural network to use 32-bit floating point weights and activations instead
of the default 64-bit. Reduces memory consumption and typically increases throughput on
modern hardware. Accept a small numerical precision trade-off in neural outputs.

### All CMake Toggles

| Flag                           | Default | Description                                      |
|--------------------------------|---------|--------------------------------------------------|
| `SELDON_ENABLE_OPENMP`         | `ON`    | Parallel neuron loops and SIMD vectorization     |
| `SELDON_ENABLE_OPENCL`         | `OFF`   | OpenCL GPU acceleration paths                   |
| `SELDON_ENABLE_CUDA`           | `OFF`   | CUDA acceleration + secondary binary target      |
| `SELDON_NEURAL_FLOAT32`        | `OFF`   | 32-bit precision neural network                  |
| `SELDON_ENABLE_NATIVE_PARQUET` | `OFF`   | Native Arrow/Parquet export backend              |

---

## Running Seldon

### Minimal Run

```bash
./build/seldon --cli /path/to/data.csv
```

Seldon will infer column types, auto-detect a target column if unspecified, run the full
pipeline, and write outputs to `<dataset_stem>_seldon_outputs/` in the current directory.

### Specify a Target Column

```bash
./build/seldon --cli /path/to/data.csv --target revenue
```

### Custom Output Directory and Report Path

```bash
./build/seldon --cli /path/to/data.csv \
  --output-dir /tmp/seldon_out \
  --report reports/model_card.md \
  --assets-dir viz
```

With relative paths, the above resolves to:

- Neural synthesis report: `/tmp/seldon_out/reports/model_card.md`
- Plot assets: `/tmp/seldon_out/viz/`

### With a Config File

```bash
./build/seldon --cli /path/to/data.csv --config analysis.json
```

CLI flags override values set in the config file.

### Full Analysis with HTML Export

```bash
./build/seldon --cli /data/input.csv \
  --output-dir /tmp/analysis \
  --generate-html true \
  --verbose-analysis true
```

### Fast Iteration Mode

```bash
./build/seldon --cli /data/input.csv --fast true
```

Caps pair-analysis breadth and model size for quick diagnostic turnaround.

### Low-Memory Execution

```bash
./build/seldon --cli /data/input.csv --low-memory true
```

Applies stricter per-stage sample limits, disables heavy bivariate operations, and reduces
plot generation. Suitable for machines with constrained heap budgets or large sparse datasets.

### Benchmark Mode

```bash
./build/seldon --cli /data/input.csv --benchmark-mode true
```

Instruments the pipeline with per-stage wall-clock timers and pair-analysis throughput
counters. Timing results are embedded in the generated reports.

---

## Output Artifacts

A complete Seldon run produces the following under the output directory:

**Markdown Reports:**

| File                    | Contents                                                       |
|-------------------------|----------------------------------------------------------------|
| `univariate.md`         | Per-column statistics, distribution summaries, outlier flags   |
| `bivariate.md`          | Pairwise correlations, MI scores, interaction diagnostics      |
| `neural_synthesis.md`   | Neural model card, feature importance, residual narrative      |
| `report.md`             | Deterministic model summary and baseline benchmark results     |
| `final_analysis.md`     | Cross-stage synthesis, causal graph, readiness matrix          |

**Plot Assets (under `seldon_report_assets/`):**

| Subdirectory | Contents                                                       |
|--------------|----------------------------------------------------------------|
| `univariate/`| Histograms, KDE plots, box plots per numeric column            |
| `bivariate/` | Scatter plots, correlation heatmaps, pair diagnostic plots     |
| `overall/`   | Feature importance bars, residual plots, model overview charts |

**Optional:**

- HTML counterparts for each Markdown report (when `--generate-html true` and `pandoc` is
  available).
- Preprocessed dataset in Parquet format (when `SELDON_ENABLE_NATIVE_PARQUET` is compiled
  in and `export_parquet` is set in config).

---

## CLI Reference

### Core Flags

| Flag                              | Description                                           |
|-----------------------------------|-------------------------------------------------------|
| `--cli <path>`                    | Path to input CSV file (required)                     |
| `--target <column>`               | Target column name for supervised analysis            |
| `--delimiter <char>`              | CSV field delimiter (default: auto-detect)            |
| `--config <path>`                 | Path to JSON configuration file                       |

### Output and Reporting

| Flag                                   | Description                                      |
|----------------------------------------|--------------------------------------------------|
| `--output-dir <path>`                  | Root output directory                            |
| `--report <path>`                      | Neural synthesis report path (abs or rel)        |
| `--assets-dir <path>`                  | Plot assets directory (abs or rel)               |
| `--generate-html true\|false`          | Convert all Markdown reports to HTML via pandoc  |
| `--store-outlier-flags-in-report true\|false` | Include outlier row flags in report body |

### Analysis Behavior

| Flag                              | Description                                           |
|-----------------------------------|-------------------------------------------------------|
| `--verbose-analysis true\|false`  | Emit extended diagnostics and intermediate summaries  |
| `--benchmark-mode true\|false`    | Instrument pipeline with per-stage timing metrics     |

### Runtime Profile

| Flag                         | Description                                                |
|------------------------------|------------------------------------------------------------|
| `--fast true\|false`         | Fast mode: capped pair analysis and lighter neural model   |
| `--low-memory true\|false`   | Low-memory mode: stricter limits, reduced plot generation  |

### Plot Controls

| Flag                             | Description                                          |
|----------------------------------|------------------------------------------------------|
| `--plot-univariate true\|false`  | Enable or disable univariate plots                   |
| `--plot-bivariate true\|false`   | Enable or disable bivariate pair plots               |
| `--plot-overall true\|false`     | Enable or disable overall summary plots              |
| `--plots <list>`                 | Comma-separated override: `univariate,bivariate,overall` |

---

## Configuration File Reference

Seldon accepts a JSON configuration file via `--config`. Any key can be overridden at
runtime by the corresponding CLI flag.

```json
{
  "dataset": "/data/input.csv",
  "target": "target_column",
  "output_dir": "/tmp/seldon_output",
  "report": "neural_synthesis.md",
  "assets_dir": "seldon_report_assets",
  "generate_html": false,
  "verbose_analysis": false,
  "store_outlier_flags_in_report": false,
  "fast_mode": false,
  "low_memory_mode": false,
  "benchmark_mode": false,
  "delimiter": ",",
  "numeric_separator_policy": "us",
  "missing_strategy": "knn",
  "outlier_method": "lof",
  "scaling": "standard",
  "export_parquet": false,
  "plot_univariate": true,
  "plot_bivariate": true,
  "plot_overall": true
}
```

**Key behaviors:**

- `numeric_separator_policy`: `"us"` (comma thousands, period decimal) or `"eu"` (period
  thousands, comma decimal).
- `missing_strategy`: `"mean"`, `"median"`, `"mode"`, or `"knn"`.
- `outlier_method`: `"lof"` (strict Local Outlier Factor) or `"iqr"`. The key
  `"lof_fallback_modified_zscore"` is a legacy alias that routes to the same strict LOF path.
- `scaling`: `"standard"` (z-score), `"minmax"`, or `"robust"` (median/IQR).

---

## Runtime Modes

Seldon supports three operational profiles, controlled by runtime flags or config keys.

### Standard Mode (default)

Full-breadth analysis: no pair caps, no model size reduction. Recommended for final analysis
runs and artifact generation.

### Fast Mode (`--fast true`)

- Bivariate pair analysis is capped to a bounded subset of feature pairs.
- Neural network topology is reduced for shorter training time.
- Plot generation proceeds normally.
- Suitable for iterative diagnostic runs during data exploration.

### Low-Memory Mode (`--low-memory true`)

- Stricter per-stage row and column sample limits.
- Heavy bivariate operations (Kendall Tau, HSIC) are disabled.
- Plot generation is reduced to summary-level outputs only.
- Heap-intensive structures (k-d tree size, matrix allocations) are bounded more tightly.
- Suitable for execution environments with constrained memory budgets or very wide datasets.

Both modes can be combined: `--fast true --low-memory true` applies all restrictions from
both profiles simultaneously.

---

## Statistical Scope and Guardrails

Seldon targets strict, textbook-defined statistical procedures for all auditable analytical
paths. The following guardrails are explicitly documented in generated reports via the
subsystem readiness matrix.

### Stable Paths

- Univariate descriptive statistics (mean, variance via Welford, percentiles, skewness,
  kurtosis) are computed with standard numerical methods and are considered stable.
- Pearson and Spearman correlations, Lasso coordinate descent feature selection, and all
  preprocessing operations (KNN imputation, LOF outlier detection, scaling) are stable.
- Deterministic baseline model results and K-fold CV metrics are stable.
- Neural network training and evaluation are stable; explainability scores are best-effort.

### Experimental Paths

- Causal discovery output should be treated as decision support. Observational causal graphs
  infer conditional independence structure; directed edge orientation is a hypothesis under CI
  test assumptions and model assumptions. It is not intervention-grade causal proof.
  Multi-algorithm consensus and bootstrap support scores are provided to indicate robustness,
  but do not eliminate confounding.
- GPU acceleration paths (OpenCL, CUDA) are environment-dependent. Validate acceleration
  effectiveness on your hardware stack before relying on throughput estimates.

### Interpretation Guidance

- Use Seldon output for prioritization, hypothesis generation, and investigation.
- Perform formal inferential validation and domain expert review before acting on high-stakes
  analytical conclusions.
- Treat causal graph directionality and LOF outlier assignments as ranked evidence, not
  ground truth.

---

## Troubleshooting

**No plots are generated.**

Check that `gnuplot` is installed and on `PATH`:

```bash
which gnuplot
gnuplot --version
```

**No HTML output despite `--generate-html true`.**

Install `pandoc` and verify it is accessible:

```bash
which pandoc
pandoc --version
```

Seldon will report the number of successful and failed HTML conversions per run. Check the
terminal output for per-file conversion status.

**OpenCL not enabled after setting `-DSELDON_ENABLE_OPENCL=ON`.**

- Run `clinfo` to verify a valid OpenCL platform and device are visible to the OS.
- Ensure the OpenCL development package is installed (headers + ICD loader). On Debian/Ubuntu:

  ```bash
  sudo apt install ocl-icd-opencl-dev opencl-headers
  ```

- Rebuild from a clean build directory:

  ```bash
  rm -rf build
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSELDON_ENABLE_OPENCL=ON
  cmake --build build -j"$(nproc)"
  ```

**Parquet export not working.**

If the binary was not compiled with `SELDON_ENABLE_NATIVE_PARQUET=ON`, Seldon will print a
runtime warning and fall back to CSV export. Rebuild with Arrow/Parquet C++ libraries
available:

```bash
sudo apt install libarrow-dev libparquet-dev   # Debian/Ubuntu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSELDON_ENABLE_NATIVE_PARQUET=ON
cmake --build build -j"$(nproc)"
```

**Very large datasets are slow.**

- Use `--fast true` to cap bivariate pair analysis breadth.
- Use `--low-memory true` to apply stricter sample limits and disable costly O(n²) tests.
- Ensure `SELDON_ENABLE_OPENMP=ON` is set in your build for parallel neural network passes.
- For the CSV parse stage, ensure the file is stored on a local fast disk (not a network
  mount), as Seldon uses memory-mapped I/O.

**Type inference misclassifies a column.**

- Use the schema override feature in the config file to force a column type explicitly.
- If a numeric column contains European-style separators (e.g. `1.234,56`), set
  `"numeric_separator_policy": "eu"` in your config.

---

## License

See [LICENSE](LICENSE).


