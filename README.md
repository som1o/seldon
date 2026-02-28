# Seldon

Seldon is an analytics engine that runs an end-to-end automated pipeline:

- data loading and preprocessing,
- deterministic + neural modeling,
- bivariate and advanced diagnostics,
- report and artifact generation.

It is designed for local batch workflows and CI-style artifact generation.

---

## 1) What Seldon Generates

For each run, Seldon creates a report output directory containing:

- `univariate.md`
- `bivariate.md`
- `final_analysis.md`
- `report.md` (deterministic report)
- neural synthesis markdown (default: `neural_synthesis.md`, configurable)
- plot assets under `seldon_report_assets/` (configurable)

Optional HTML exports are generated when `generate_html=true` and `pandoc` is available.

---

## 2) Prerequisites

- Linux/macOS with a C++17 compiler (GCC/Clang)
- CMake >= 3.10
- Optional:
  - OpenMP runtime/dev package
  - OpenCL ICD + headers (`OpenCL`) for OpenCL build path
  - `gnuplot` for plotting
  - `pandoc` for HTML report export

---

## 3) Build

### 3.1 Clean Release build (OpenCL enabled)

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSELDON_ENABLE_OPENCL=ON -DSELDON_ENABLE_OPENMP=ON
cmake --build build -j"$(nproc)"
```

If OpenCL is found, CMake prints:

- `Seldon: OpenCL enabled`

If not found, Seldon builds normally without OpenCL acceleration hooks.

### 3.2 Common build toggles

- `-DSELDON_ENABLE_OPENMP=ON|OFF`
- `-DSELDON_ENABLE_OPENCL=ON|OFF`
- `-DSELDON_NEURAL_FLOAT32=ON|OFF`
- `-DSELDON_ENABLE_CUDA=ON|OFF` (optional extra target if toolkit is available)

---

## 4) Run

### 4.0 GUI Dashboard (default)

Seldon now launches a GTK4 desktop dashboard by default:

```bash
./build/seldon
```

The dashboard exposes core pipeline controls in dedicated tabs and includes:

- `Extra CLI flags` input (supports any existing CLI switch),
- `Config overlay` input (supports any config key via `key: value` lines),

so every CLI/config capability remains accessible from the GUI.

### 4.0.1 CLI fallback mode

Use CLI mode explicitly with:

```bash
./build/seldon --cli /path/to/data.csv [other flags]
```

### 4.1 Minimal

```bash
./build/seldon --cli /path/to/data.csv
```

### 4.2 Custom output + report paths

```bash
./build/seldon --cli /path/to/data.csv \
  --output-dir /tmp/seldon_out \
  --report custom_neural_report.md \
  --assets-dir custom_assets
```

Notes:

- `--output-dir` sets the root output directory.
- `--report` and `--assets-dir` accept absolute or relative paths.
- Relative `--report`/`--assets-dir` values are resolved under `--output-dir`.
- Generated markdown links/images are normalized relative to each report file location for portability across filesystem roots.

### 4.3 With config file

```bash
./build/seldon --cli /path/to/data.csv --config config.json
```

---

## 5) Reporting Behavior (Current)

- If `output_dir` is not provided, Seldon creates `<dataset_stem>_seldon_outputs` beside the dataset.
- If `report` is unset (or default), neural synthesis is written to `<output_dir>/neural_synthesis.md`.
- If `assets_dir` is unset (or default), assets go to `<output_dir>/seldon_report_assets`.
- HTML export path for neural synthesis follows the configured report filename stem.

---

## 6) Useful Runtime Flags

- `--target <column>`
- `--output-dir <path>`
- `--report <path>`
- `--assets-dir <path>`
- `--generate-html true|false`
- `--verbose-analysis true|false`
- `--store-outlier-flags-in-report true|false`
- `--fast true|false`
- `--low-memory true|false`

See `docs/USAGE.md` for full parameter coverage.

---

## 7) Docs

- Usage and config reference: `docs/USAGE.md`
- Pipeline/component design: `docs/ARCHITECTURE.md`

---

## 8) Troubleshooting

- **No plots generated**: check `gnuplot` in PATH.
- **No HTML generated**: install `pandoc` and set `generate_html=true`.
- **OpenCL expected but not enabled**:
  - verify `clinfo` output,
  - install OpenCL development package (`OpenCLConfig.cmake` / headers + ICD),
  - rebuild with `-DSELDON_ENABLE_OPENCL=ON`.

---

## 9) Statistical Scope and Heuristic Components

Seldon now prioritizes strict, textbook-style statistical procedures for core analytical paths (including `outlier_method=lof` using strict Local Outlier Factor scoring instead of heuristic fallback logic).

Examples:

- Causal discovery output is produced through observational constraint/score-based methods with bootstrap support, and remains assumption-dependent.
- `outlier_method=lof` runs strict LOF; `lof_fallback_modified_zscore` is kept as a legacy alias and routes to the same strict LOF path.
- Directional edges in observational data are hypotheses under CI/model assumptions, not intervention-level causal proof.

Use this output for prioritization and investigation; perform formal inferential validation when decisions are high-stakes.
