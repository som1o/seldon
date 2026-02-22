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
   - `univariate.txt`
   - `bivariate.txt`
   - `neural_synthesis.txt`
   - `final_analysis.txt`
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
- `NeuralNet`: feed-forward network with deterministic seed + gradient clipping
- `GnuplotEngine`: optional PNG plotting backend
- `ReportEngine`: plain-text report writer

## Design Notes
- `TypedDataset` is the main production dataset representation.
- Pearson correlation in pipeline delegates to `MathUtils` to avoid duplicated statistical formulas.
- Plot generation gracefully degrades when `gnuplot` is unavailable; analysis reports are still fully generated.

## Determinism & Numerical Safety
- Neural training supports fixed seed control (`neural_seed`) for reproducibility.
- Global gradient clipping (`gradient_clip_norm`) mitigates unstable updates.
- Statistical significance is computed consistently through `MathUtils`.

## Extension Points
- Replace `GnuplotEngine` with another plotting backend while preserving report contract.
- Introduce richer model ensembles in `BenchmarkEngine`.
- Expand typed datetime feature engineering in `TypedDataset` + preprocessing stage.
