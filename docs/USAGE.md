# Usage Guide

## Build (with OpenMP)
```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

Disable OpenMP explicitly when needed:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
```

## Basic Run
```bash
./seldon /path/to/data.csv
```

## Supervised Plot Modes
Plots are opt-in by mode.

```bash
./seldon /path/to/data.csv --plots bivariate
./seldon /path/to/data.csv --plots all
./seldon /path/to/data.csv --plot-univariate true --plot-overall false --plot-bivariate true
./seldon /path/to/data.csv --generate-html true
```

## Verbose Analysis
```bash
./seldon /path/to/data.csv --verbose-analysis true
```

## Neural Stability & Reproducibility
```bash
./seldon /path/to/data.csv --neural-seed 1337 --benchmark-seed 1337 --gradient-clip 5.0
```

Skip neural training entirely (correlation-driven relevance only):

```bash
./seldon /path/to/data.csv --neural-strategy none
```

## Fast Mode (Large Datasets)
Use fast mode for very large datasets (for example, >100k rows or >50 numeric columns):

```bash
./seldon /path/to/data.csv --fast true --fast-max-bivariate-pairs 2500 --fast-neural-sample-rows 25000
```

What fast mode changes:
- forces neural policy to `fast`
- caps bivariate pair evaluation
- samples rows for neural training only (full data is still used for ingestion/preprocessing/statistics)

## Heuristic Tuning (Advanced)
```bash
./seldon /path/to/data.csv \
	--feature-min-variance 1e-10 \
	--feature-leakage-corr-threshold 0.995 \
	--significance-alpha 0.05 \
	--outlier-iqr-multiplier 1.5 \
	--outlier-z-threshold 3.0 \
	--bivariate-selection-quantile 0.65
```

## Outputs
Always generated:
- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`

Optional (when `--generate-html true` and `pandoc` is available):
- `univariate.html`
- `bivariate.html`
- `neural_synthesis.html`
- `final_analysis.html`

Plots (when enabled) are generated as PNG only in:
- `seldon_report_assets/univariate`
- `seldon_report_assets/bivariate`
- `seldon_report_assets/overall`

## Fallback Behavior
If `gnuplot` is unavailable, Markdown reports are still generated with full statistical analysis; only plot images are skipped.

## Datetime Parsing
Accepted datetime forms include:
- `YYYY-MM-DD`
- `YYYY-MM-DD HH:MM:SS`
- `MM/DD/YYYY`
- `DD-MM-YYYY`

If a detected datetime column contains too many non-missing values that do not parse (mixed/unexpected formats), Seldon falls back to categorical for that column.

## Report Portability
Markdown image links are written as relative paths when possible to improve report portability.
