# Seldon Usage Guide

This guide focuses on practical build/run usage for the current Seldon pipeline.

---

## 1) Build and Run Quickstart

### 1.1 Clean Release build with OpenCL

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSELDON_ENABLE_OPENCL=ON -DSELDON_ENABLE_OPENMP=ON
cmake --build build -j"$(nproc)"
```

Expected configure output when available:

- `Seldon: OpenCL enabled`

### 1.2 Run

```bash
./build/seldon /path/to/data.csv
```

---

## 2) Output and Reporting Paths

### 2.1 Defaults

- Output root: `<dataset_stem>_seldon_outputs`
- Neural synthesis markdown: `<output_root>/neural_synthesis.md`
- Assets root: `<output_root>/seldon_report_assets`

### 2.2 Overrides

CLI flags:

- `--output-dir <path>`
- `--report <path>`
- `--assets-dir <path>`

Path rules:

- absolute paths are used as-is,
- relative `report`/`assets_dir` are resolved under `output_dir`.

Example:

```bash
./build/seldon data.csv \
  --output-dir /tmp/seldon_out \
  --report reports/model_card.md \
  --assets-dir viz
```

Result:

- report => `/tmp/seldon_out/reports/model_card.md`
- assets => `/tmp/seldon_out/viz`

---

## 3) Key CLI Options

Core:

- `--target <column>`
- `--delimiter <char>`
- `--config <path>`
- `--verbose-analysis true|false`

Reporting/outputs:

- `--output-dir <path>`
- `--report <path>`
- `--assets-dir <path>`
- `--generate-html true|false`
- `--store-outlier-flags-in-report true|false`

Runtime profile:

- `--fast true|false`
- `--low-memory true|false`

Plot controls:

- `--plots <bivariate,univariate,overall>`
- `--plot-univariate true|false`
- `--plot-overall true|false`
- `--plot-bivariate true|false`

---

## 4) Config File Keys (Common)

```yaml
dataset: /data/input.csv
target: target_column
output_dir: /tmp/seldon_output
report: neural_synthesis.md
assets_dir: seldon_report_assets
generate_html: false
verbose_analysis: false
store_outlier_flags_in_report: false
fast_mode: false
low_memory_mode: false
```

You can still override these via CLI flags.

---

## 5) Build Variants

### 5.1 OpenMP only

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSELDON_ENABLE_OPENMP=ON -DSELDON_ENABLE_OPENCL=OFF
cmake --build build -j"$(nproc)"
```

### 5.2 OpenCL + OpenMP

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSELDON_ENABLE_OPENMP=ON -DSELDON_ENABLE_OPENCL=ON
cmake --build build -j"$(nproc)"
```

### 5.3 Optional CUDA target

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSELDON_ENABLE_CUDA=ON
cmake --build build -j"$(nproc)"
```

If CUDA is available, `seldon_cuda` is built in addition to `seldon`.

---

## 6) Typical Workflows

### 6.1 Full analysis with HTML export

```bash
./build/seldon /data/input.csv \
  --output-dir /tmp/seldon_output \
  --generate-html true \
  --verbose-analysis true
```

### 6.2 Fast iteration mode

```bash
./build/seldon /data/input.csv --fast true --verbose-analysis false
```

### 6.3 Low-memory execution

```bash
./build/seldon /data/input.csv --low-memory true
```

---

## 7) Generated Artifacts

Markdown:

- `univariate.md`
- `bivariate.md`
- `final_analysis.md`
- `report.md`
- configured neural synthesis markdown (`report`)

Assets:

- `<assets_dir>/univariate`
- `<assets_dir>/bivariate`
- `<assets_dir>/overall`

Optional HTML:

- `univariate.html`
- `bivariate.html`
- HTML pair of configured neural synthesis report path
- `final_analysis.html`
- `report.html`

---

## 8) Troubleshooting

### OpenCL not enabled

- Check `clinfo`
- Install OpenCL headers + ICD loader/dev package
- Reconfigure with `-DSELDON_ENABLE_OPENCL=ON`

### No plots in reports

- Install `gnuplot`
- Verify plot flags are enabled

### No HTML files

- Install `pandoc`
- Set `--generate-html true`

---

## 9) Heuristic Components and Interpretation

Some Seldon modules are intentionally heuristic for runtime practicality. Treat these as discovery aids, not rigorous statistical proof.

- Causal graph candidates are heuristic/constraint-guided and require domain validation.
- `outlier_method=lof` (or `lof_fallback_modified_zscore`) uses a modified Z-score fallback heuristic rather than strict LOF.

For high-stakes use cases, validate findings with formal statistical or experimental methods.
