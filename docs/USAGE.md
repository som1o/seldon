# Usage Guide

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
```

## Verbose Analysis
```bash
./seldon /path/to/data.csv --verbose-analysis true
```

## Neural Stability & Reproducibility
```bash
./seldon /path/to/data.csv --neural-seed 1337 --gradient-clip 5.0
```

## Outputs
Always generated:
- `univaraite.txt`
- `bivariate.txt`
- `neural_synthesis.txt`
- `final_analysis.txt`

Plots (when enabled) are generated as PNG only in:
- `seldon_report_assets/univariate`
- `seldon_report_assets/bivariate`
- `seldon_report_assets/overall`

## Fallback Behavior
If `gnuplot` is unavailable, text reports are still generated with full statistical analysis; only plot images are skipped.
