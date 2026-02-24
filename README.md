# Seldon

Seldon is a C++17 tabular analysis pipeline that runs ingestion, preprocessing, statistics, neural analysis, and reporting in one command.

It supports mixed-type datasets (`numeric`, `categorical`, `datetime`) and now handles compressed and Excel sources directly.

## Key Capabilities

- Typed ingestion with robust parsing for mixed schemas.
- Input formats: `.csv`, `.csv.gz`, `.csv.zip`, `.xlsx`, `.xls`.
- Preprocessing with missing-data handling, outlier treatment, scaling, and feature engineering.
- Univariate + bivariate statistical analysis with optional plotting.
- Neural relevance analysis with adaptive architecture, multi-output support, uncertainty, and explainability.
- Markdown report synthesis with optional HTML export.
- Profile presets and interactive setup mode for faster configuration.

## Requirements

Core:

- C++17 compiler (`g++`/`clang++`)
- CMake 3.16+

Optional:

- OpenMP runtime (parallel execution)
- `gnuplot` (plot generation)
- `pandoc` (HTML reports)
- `xlsx2csv` for `.xlsx`
- `xls2csv` for `.xls`
- `gzip` / `unzip` for compressed CSV inputs

Ubuntu/Debian example for Excel converters:

```bash
sudo apt-get update
sudo apt-get install -y xlsx2csv catdoc
```

## Build

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

OpenMP-disabled build:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
cmake --build . -j
```

## Quick Start

From `build/`:

```bash
./seldon /path/to/data.csv
```

Using compressed or Excel input:

```bash
./seldon /path/to/data.csv.gz
./seldon /path/to/data.xlsx
```

Using a config file:

```bash
./seldon /path/to/data.csv --config /path/to/config.yaml
```

Interactive setup:

```bash
./seldon --interactive
```

## Profiles and Runtime Modes

Profiles quickly tune behavior:

```bash
./seldon /path/to/data.csv --profile quick
./seldon /path/to/data.csv --profile thorough
./seldon /path/to/data.csv --profile minimal
```

Supported profile values: `auto`, `quick`, `thorough`, `minimal`.

## Neural Analysis Features

- Streaming / incremental training (`neural_streaming_mode`).
- Adaptive depth/width search (`neural_min_layers`, `neural_max_layers`, `neural_max_hidden_nodes`).
- Multi-output with auxiliary targets (`neural_multi_output`, `neural_max_aux_targets`).
- Explainability modes: `permutation`, `integrated_gradients`, `shap_approx`, `hybrid`.
- Monte Carlo uncertainty (`neural_uncertainty_samples`).
- Feature-importance controls (`neural_importance_max_rows`, `neural_importance_trials`, `neural_importance_parallel`).

Example:

```bash
./seldon /path/to/data.csv \
	--neural-explainability hybrid \
	--neural-integrated-grad-steps 24 \
	--neural-uncertainty-samples 32 \
	--neural-importance-max-rows 600 \
	--neural-importance-trials 1
```

## Output Structure

If `--output-dir` is not provided, Seldon writes output near the dataset as:

`<dataset_stem>_seldon_outputs/`

Inside the output folder:

- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`
- `seldon_report_assets/`

Optional HTML reports are generated when `generate_html=true` and `pandoc` is available.

## Preprocessed Dataset Export

Export cleaned/transformed data:

```bash
./seldon /path/to/data.csv --export-preprocessed csv
./seldon /path/to/data.csv --export-preprocessed parquet
./seldon /path/to/data.csv --export-preprocessed csv --export-preprocessed-path /tmp/my_preprocessed
```

Config keys:

- `export_preprocessed: none|csv|parquet`
- `export_preprocessed_path: /path/base_name`

## Minimal Config Example

```yaml
dataset: /path/to/data.xlsx
target: sales
profile: quick
plots: bivariate
datetime_locale_hint: auto
numeric_locale_hint: auto
export_preprocessed: none
```

## Documentation

- [Usage Reference](docs/USAGE.md)
- [Architecture Reference](docs/ARCHITECTURE.md)
- [Excel Import Enablement](docs/ENABLE_EXCEL_IMPORT.md)

## License

MIT License. See [LICENSE](LICENSE).
