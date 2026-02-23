# Seldon Architecture

## Overview
Seldon is organized as a deterministic analytics pipeline centered on `TypedDataset`.

1. **Ingestion**: CSV parsing + typed inference (`numeric`, `categorical`, `datetime`)
2. **Preprocessing**: missing handling, outlier processing, scaling
3. **Statistics & Decisioning**:
   - Univariate statistical profiling
   - Bivariate significance analysis
   - Neural-lattice relevance scoring
4. **Reporting**:
   - `univariate.md`
   - `bivariate.md`
   - `neural_synthesis.md`
   - `final_analysis.md`
   - Optional self-contained HTML counterparts via `pandoc` when `generate_html=true`
5. **Plotting (Supervised)**:
   - `seldon_report_assets/univariate`
   - `seldon_report_assets/bivariate`
   - `seldon_report_assets/overall`

## Core Modules
- `AutomationPipeline`: end-to-end orchestration
- `TypedDataset`: typed storage and row-aligned transformations
- `CSVUtils`: shared CSV line parser/BOM/header normalization
- `Preprocessor`: missing/outlier/scaling transformations
- `MathUtils`: statistical significance and matrix math
- `NeuralNet`: feed-forward network with deterministic seed, GELU hidden activation, batch normalization, post-activation layer normalization, adaptive LR-on-plateau decay, Lookahead optimizer support, and gradient clipping
- `GnuplotEngine`: optional PNG plotting backend
- `ReportEngine`: markdown report writer

## Design Notes
- `TypedDataset` is the main production dataset representation.
- Pearson correlation in pipeline delegates to `MathUtils` to avoid duplicated statistical formulas.
- Numeric `ColumnStats` are cached post-preprocessing and reused across univariate/bivariate/overall analysis stages.
- Plot generation gracefully degrades when `gnuplot` is unavailable; analysis reports are still fully generated.
- Bivariate pair scoring uses OpenMP parallel loops in non-verbose mode.
- Neural stage can be bypassed with `neural_strategy=none`.

## Determinism & Numerical Safety
- Neural training supports fixed seed control (`neural_seed`) for reproducibility.
- GELU hidden activations reduce dead-neuron behavior compared with plain ReLU.
- Hidden-layer batch normalization re-centers activations with running mean/variance.
- Hidden-layer post-activation layer normalization dampens variance drift across lattice states.
- Lookahead optimizer keeps slow-moving shadow weights synchronized from fast updates for jitter reduction.
- Validation-plateau scheduler monitors EMA-smoothed validation loss, applies cooldown between LR cuts, and caps reductions per run (floor at `1e-6`).
- Gradient clipping (`gradient_clip_norm`) applies both element-wise clamping and global norm scaling.
- Backprop uses pre-activation derivatives and explicit dropout scaling to keep gradient flow mathematically consistent across activations.
- Early stopping restores best validation checkpointed weights/biases before inference/reporting.
- Statistical significance is computed consistently through `MathUtils`.
- Incomplete beta fallback controls are configurable (`beta_fallback_*`, `numeric_epsilon`).

## Extension Points
- Replace `GnuplotEngine` with another plotting backend while preserving report contract.
- Introduce richer model ensembles in `BenchmarkEngine`.
- Expand typed datetime feature engineering in `TypedDataset` + preprocessing stage.
