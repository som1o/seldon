# Usage Reference

## 1. Build

### 1.1 Requirements

- C++17 compiler
- CMake 3.16+
- Optional: OpenMP runtime
- Optional: `gnuplot` for plots
- Optional: `pandoc` for HTML export

### 1.2 Build Commands

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

Disable OpenMP if needed:

```bash
cmake -DSELDON_ENABLE_OPENMP=OFF ..
cmake --build . -j
```

## 2. Supported Input Sources

Seldon accepts:

- `.csv`
- `.csv.gz` (via `gzip -cd`)
- `.csv.zip` (via `unzip -p`)
- `.xlsx` (via `xlsx2csv`)
- `.xls` (via `xls2csv`)

### 2.1 Excel Enablement

Install converters on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y xlsx2csv catdoc
```

If converter tools are missing, Seldon returns a clear runtime error indicating what to install.

See also: `docs/ENABLE_EXCEL_IMPORT.md`.

## 3. Running Seldon

Minimal run:

```bash
./seldon /path/to/data.csv
```

Run with config:

```bash
./seldon /path/to/data.csv --config /path/to/config.yaml
```

Interactive config wizard:

```bash
./seldon --interactive
```

Profile-driven run:

```bash
./seldon /path/to/data.csv --profile quick
./seldon /path/to/data.csv --profile thorough
./seldon /path/to/data.csv --profile minimal
```

Locale hints:

```bash
./seldon /path/to/data.csv --datetime-locale-hint dmy --numeric-locale-hint eu
```

## 4. Output Layout

By default, output is written next to the dataset in a folder named:

`<dataset_stem>_seldon_outputs/`

Use a custom output folder with:

```bash
./seldon /path/to/data.csv --output-dir /path/to/output_folder
```

Main artifacts inside the output folder:

- `univariate.md`
- `bivariate.md`
- `neural_synthesis.md`
- `final_analysis.md`
- `seldon_report_assets/` (plot assets)

Optional HTML artifacts are generated when `generate_html=true` and `pandoc` is available.

## 5. Preprocessed Dataset Export

Export preprocessed data:

```bash
./seldon /path/to/data.csv --export-preprocessed csv
./seldon /path/to/data.csv --export-preprocessed parquet
```

Custom export base path:

```bash
./seldon /path/to/data.csv --export-preprocessed csv --export-preprocessed-path /tmp/my_preprocessed
```

Config values:

- `export_preprocessed: none|csv|parquet`
- `export_preprocessed_path: /path/base_name`

If parquet conversion tooling is unavailable, Seldon keeps the CSV export and logs guidance.

## 6. Configuration File (`key: value`)

Minimal example:

```yaml
dataset: /path/to/data.xlsx
target: sales
profile: quick
plots: bivariate
datetime_locale_hint: auto
numeric_locale_hint: auto
export_preprocessed: none
```

Extended example:

```yaml
dataset: /path/to/data.csv.gz
target: sales
delimiter: ,

profile: thorough
output_dir: /tmp/seldon_sales_outputs
plots: all
plot_theme: auto
generate_html: false
verbose_analysis: true

datetime_locale_hint: dmy
numeric_locale_hint: eu

export_preprocessed: csv
export_preprocessed_path: /tmp/sales_preprocessed

neural_streaming_mode: true
neural_streaming_chunk_rows: 2048
neural_multi_output: true
neural_max_aux_targets: 2
neural_explainability: hybrid
neural_integrated_grad_steps: 24
neural_uncertainty_samples: 32
neural_importance_parallel: true
neural_importance_max_rows: 800
neural_importance_trials: 2

feature_engineering_enable_poly: true
feature_engineering_enable_log: true
feature_engineering_degree: 3
feature_engineering_max_base: 10
```

## 7. High-Impact Options

Data/control:

- `profile`: `auto|quick|thorough|minimal`
- `datetime_locale_hint`: `auto|dmy|mdy`
- `numeric_locale_hint`: `auto|us|eu`
- `target`, `delimiter`, `exclude`, `impute.<column>`

Neural/performance:

- `neural_streaming_mode`, `neural_streaming_chunk_rows`
- `neural_multi_output`, `neural_max_aux_targets`
- `neural_explainability`: `permutation|integrated_gradients|shap_approx|hybrid`
- `neural_importance_max_rows`, `neural_importance_trials`, `neural_importance_parallel`
- `fast_mode`, `fast_max_bivariate_pairs`, `fast_neural_sample_rows`

Output/export:

- `output_dir`
- `export_preprocessed`, `export_preprocessed_path`
- `report`, `assets_dir`, `generate_html`

## 8. Troubleshooting

### 8.1 Feature Importance Is Slow

Use lower caps first:

```bash
./seldon /path/to/data.csv --neural-importance-max-rows 600 --neural-importance-trials 1 --profile quick
```

### 8.2 Excel or Compressed Input Fails

Check converter availability:

```bash
command -v xlsx2csv
command -v xls2csv
command -v gzip
command -v unzip
```

### 8.3 Missing Plot Outputs

Ensure `gnuplot` is in `PATH` and plotting is enabled.

### 8.4 Missing HTML Outputs

Ensure `pandoc` is in `PATH` and `generate_html=true`.

- [Project Overview](../README.md)
- [Architecture Reference](ARCHITECTURE.md)
