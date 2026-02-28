# Seldon Architecture

This document describes the current runtime architecture and component boundaries.

---

## 1) High-Level Pipeline

`AutomationPipeline::run` orchestrates the full flow:

1. Resolve runtime config and output paths
2. Cleanup output/artifact directories
3. Load typed dataset
4. Target resolution + preprocessing
5. Univariate profiling
6. Feature selection + deterministic guards
7. Benchmarks + neural analysis
8. Bivariate and advanced analytics
9. Report assembly
10. Artifact save + completion summary

Main runtime entry:

- `src/pipeline_parts/PipelineRuntime.cpp`

---

## 2) Pipeline Part Layout

The pipeline is split into focused implementation units included by `src/AutomationPipeline.cpp`:

- `src/pipeline_parts/PipelineUnivariate.cpp`
  - utility scaffolding, plotting helpers, I/O/report save helpers, path cleanup, target/feature selection helpers
- `src/pipeline_parts/PipelineModeling.cpp`
  - neural training strategies, explainability, bivariate scoring policy, feature approval logic
- `src/pipeline_parts/PipelineBivariate.cpp`
  - pair analytics, advanced diagnostics (interaction/drift/context)
- `src/pipeline_parts/PipelineReporting.cpp`
  - residual narrative and outlier contextualization helpers
- `src/pipeline_parts/PipelineRuntime.cpp`
  - end-to-end execution flow and report composition

---

## 3) Core Engines

- `TypedDataset` (`include/TypedDataset.h`, `src/TypedDataset.cpp`)
  - typed columns, missing masks, schema inference and overrides
- `Preprocessor` (`include/Preprocessor.h`, `src/Preprocessor.cpp`)
  - missing handling, outlier handling, scaling, preprocess report
- `Statistics` / `MathUtils`
  - robust descriptive and inferential statistics
- `NeuralNet` (`include/NeuralNet.h`, `src/NeuralNet.cpp`)
  - dense network, optimizers, explainability hooks, uncertainty estimates
- `BenchmarkEngine`
  - baseline model family and CV summaries
- `ReportEngine`
  - markdown report rendering and persistence

---

## 4) Output Path Resolution Rules

Runtime path behavior is centralized in `PipelineRuntime.cpp`:

- If `outputDir` empty: derive `<dataset_stem>_seldon_outputs`
- `assetsDir`:
  - default/empty => `<outputDir>/seldon_report_assets`
  - relative => `<outputDir>/<assetsDir>`
  - absolute => unchanged
- `reportFile`:
  - default/empty => `<outputDir>/neural_synthesis.md`
  - relative => `<outputDir>/<reportFile>`
  - absolute => unchanged

HTML neural output path uses configured report file stem (not hardcoded).

---

## 5) Build and Acceleration Model

Build system: CMake (`CMakeLists.txt`)

Compile-time toggles:

- `SELDON_ENABLE_OPENMP`
- `SELDON_ENABLE_OPENCL`
- `SELDON_NEURAL_FLOAT32`
- `SELDON_ENABLE_CUDA`

Behavior:

- OpenMP and OpenCL are optional and conditionally linked when discovered.
- CUDA can produce a secondary `seldon_cuda` binary when toolkit is available.

---

## 6) Reporting Stack

Primary reports emitted per run:

- `univariate.md`
- `bivariate.md`
- configured neural synthesis markdown (`reportFile`)
- `final_analysis.md`
- `report.md`

Plot assets live under `assetsDir` with subfolders:

- `univariate`
- `bivariate`
- `overall`

Optional HTML conversion is performed via `pandoc` if enabled.

---

## 7) Runtime Modes

- Standard mode: full breadth analysis
- `fast_mode`: pair caps/sample limits for quicker iteration
- `low_memory_mode`: stricter limits, reduced heavy operations, plotting reduction

These are applied early in `AutomationPipeline::run` and propagate to downstream stages.

---

## 8) Design Notes

- Keep data processing deterministic first, neural second.
- Keep reporting path behavior config-driven and predictable.
- Keep optional acceleration (OpenMP/OpenCL/CUDA) non-blocking: missing backends should degrade gracefully.
- Some analytics modules are intentionally heuristic (including causal discovery guidance and LOF-labeled outlier fallback), and should be interpreted as decision support rather than formal causal/statistical proof.
