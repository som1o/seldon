# Seldon

Seldon is a C++17 tabular analytics and reporting system.

It loads heterogeneous datasets,
infers and manages column types,
applies robust preprocessing,
runs statistical and neural analysis,
and emits structured reports in one execution flow.

This document is intentionally detailed.

It is designed as the primary onboarding reference for:

- data practitioners,
- platform engineers,
- ML engineers,
- QA users,
- and operators running batch analytics jobs.

---

## 1) What Seldon Does

Seldon provides one integrated runtime pipeline for:

- dataset ingestion,
- typed interpretation of columns,
- missing-value handling,
- outlier handling,
- feature engineering,
- statistical profiling,
- bivariate significance screening,
- neural relevance analysis,
- benchmark comparison,
- and report generation.

Seldon is designed for tabular datasets with mixed schemas.

Supported logical column classes:

- numeric,
- categorical,
- datetime.

Primary outcomes:

- `univariate.md`,
- `bivariate.md`,
- `neural_synthesis.md`,
- `final_analysis.md`,
- plot assets,
- optional HTML artifacts.

---

## 2) Supported Input Formats

Seldon accepts the following input paths directly:

- `.csv`,
- `.csv.gz`,
- `.csv.zip`,
- `.xlsx`,
- `.xls`.

Format-specific notes:

- `.csv.gz` is streamed via `gzip -cd`.
- `.csv.zip` is streamed via `unzip -p`.
- `.xlsx` is converted via `xlsx2csv`.
- `.xls` is converted via `xls2csv`.

When optional converters are unavailable,
Seldon fails with explicit operational messages.

---

## 3) Core Characteristics

Seldon is built around the following principles:

- deterministic pipeline staging,
- typed column alignment,
- configuration-first runtime behavior,
- robust numeric safeguards,
- practical defaults with explicit overrides,
- artifact-oriented output.

Additional practical properties:

- OpenMP acceleration when enabled,
- fallback serial execution when disabled,
- file-based reports suitable for CI artifacts,
- command-line operation suitable for automation.

---

## 4) Build Requirements

### 4.1 Required

- C++17 compiler,
- CMake 3.16+.

### 4.2 Runtime Packages (Recommended)

- OpenMP runtime,
- `gnuplot`,
- `pandoc`,
- `xlsx2csv`,
- `xls2csv`,
- `gzip`,
- `unzip`.

Ubuntu/Debian quick install:

```bash
sudo apt update
sudo apt install -y gnuplot pandoc xlsx2csv catdoc gzip unzip
```

`xls2csv` is provided by `catdoc` on Debian/Ubuntu.

---

## 5) Build Commands

### 5.1 Standard Build

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON ..
cmake --build . -j
```

### 5.2 Build Without OpenMP

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=OFF ..
cmake --build . -j
```

### 5.3 Build With Neural Float32

Use this to cut neural tensor memory roughly in half (weights/activations/gradients).

```bash
mkdir -p build
cd build
cmake -DSELDON_ENABLE_OPENMP=ON -DSELDON_NEURAL_FLOAT32=ON ..
cmake --build . -j
```

### 5.4 Binary Location

The executable is emitted as:

- `build/seldon`

### 5.5 Memory Profiling With Massif

If `valgrind` is installed, profile peak memory with:

```bash
valgrind --tool=massif --massif-out-file=massif.out \
  ./build/seldon /absolute/path/to/data.csv --low-memory true

ms_print massif.out > massif_report.txt
```

Focus on the largest snapshots in `massif_report.txt` and compare before/after enabling:

- `--low-memory true`
- `--neural-streaming-mode true`
- `-DSELDON_NEURAL_FLOAT32=ON`

---

## 6) First Run

From `build/`:

```bash
./seldon /absolute/path/to/data.csv
```

With explicit target and no plots:

```bash
./seldon /absolute/path/to/data.csv --target sales --plots none
```

With automatic target strategy and profile:

```bash
./seldon /absolute/path/to/data.csv --target-strategy auto --profile quick
```

---

## 7) Execution Modes

Seldon supports multiple workflow styles.

### 7.1 Direct CLI Mode

Pass dataset and options directly.

### 7.2 Config-Backed Mode

Pass a config file and optionally override keys from CLI.

```bash
./seldon /absolute/path/to/data.csv --config /absolute/path/to/seldon.yaml
```

### 7.3 Interactive Wizard

Generate a config interactively:

```bash
./seldon --interactive
```

---

## 8) Runtime Profiles

Profile values:

- `auto`,
- `quick`,
- `thorough`,
- `minimal`.

Profile intent:

- `quick`: prioritize speed.
- `thorough`: prioritize analytical depth.
- `minimal`: produce lean outputs quickly.
- `auto`: infer suitable mode from data scale.

Examples:

```bash
./seldon /path/data.csv --profile quick
./seldon /path/data.csv --profile thorough
./seldon /path/data.csv --profile minimal
```

---

## 9) Locale and Parsing Hints

Date parsing hint:

- `datetime_locale_hint=auto|dmy|mdy`

Numeric separator hint:

- `numeric_locale_hint=auto|us|eu`

CLI examples:

```bash
./seldon /path/data.csv --datetime-locale-hint dmy --numeric-locale-hint eu
./seldon /path/data.csv --datetime-locale-hint mdy --numeric-locale-hint us
```

---

## 10) Explicit Type Overrides

Seldon infers types automatically,
and also supports explicit overrides.

CLI form:

- `--type <column>:numeric`
- `--type <column>:categorical`
- `--type <column>:datetime`

Examples:

```bash
./seldon /path/data.csv --type invoice_date:datetime --type segment:categorical
./seldon /path/data.csv --type amount:numeric
```

Config form:

- `type.invoice_date: datetime`
- `type.segment: categorical`
- `type.amount: numeric`

Type overrides are applied during load,
before fallback reinterpretation logic.

---

## 11) Missing Data and Imputation

Seldon supports per-column imputation directives.

Allowed strategies:

- `auto`,
- `mean`,
- `median`,
- `zero`,
- `mode`,
- `interpolate`.

Config example:

```yaml
impute.amount: median
impute.city: mode
impute.event_time: interpolate
```

Rules are validated against column type compatibility.

---

## 12) Outlier Handling

Detection methods:

- `iqr`,
- `zscore`,
- `modified_zscore`,
- `adjusted_boxplot`,
- `lof`.

Actions:

- `flag`,
- `remove`,
- `cap`.

Memory-related behavior:

- full row-level outlier flags are optional,
- controlled by `store_outlier_flags_in_report`.

This avoids retaining large boolean maps unless needed.

---

## 13) Scaling

Scaling options:

- `auto`,
- `zscore`,
- `minmax`,
- `none`.

Scaling is applied to numeric features.

Binary targets are preserved safely when identified as binary series.

---

## 14) Feature Engineering Controls

Switches:

- `feature_engineering_enable_poly`,
- `feature_engineering_enable_log`,
- `feature_engineering_enable_ratio_product_discovery`.

Degree and base limits:

- `feature_engineering_degree`,
- `feature_engineering_max_base`,
- `feature_engineering_max_pairwise_discovery`.

Expansion guard:

- `feature_engineering_max_generated_columns`.

The generated-column cap is now explicit and enforced.

---

## 15) Neural Analysis Overview

Neural analysis includes:

- adaptive topology,
- optimizer controls,
- dropout,
- normalization,
- LR scheduling,
- early stopping,
- gradient clipping,
- explainability,
- uncertainty estimation.

Reliability guardrails now include:

- Monte Carlo dropout uncertainty,
- mini-ensemble disagreement,
- OOD distance scoring,
- PSI-based drift sentinels.

Key controls:

- `neural_strategy`,
- `neural_learning_rate`,
- `neural_optimizer`,
- `neural_lookahead_fast_optimizer`,
- `neural_lookahead_sync_period`,
- `neural_lookahead_alpha`,
- `gradient_clip_norm`,
- `neural_uncertainty_samples`,
- `neural_ensemble_members`,
- `neural_ensemble_probe_rows`,
- `neural_ensemble_probe_epochs`,
- `neural_ood_enabled`,
- `neural_ood_z_threshold`,
- `neural_ood_distance_threshold`,
- `neural_drift_psi_warning`,
- `neural_drift_psi_critical`.

---

## 16) Neural Safety Limits

Seldon enforces user-configurable topology safety limits:

- `neural_max_topology_nodes`,
- `neural_max_trainable_params`.

Seldon also enforces hard constructor-level safety guards
inside the neural model implementation.

These protections prevent pathological memory allocations
from hostile or accidental oversized settings.

---

## 17) Categorical Encoding Controls

Categorical features are one-hot encoded with caps.

Key control:

- `neural_max_one_hot_per_column`.

When categories exceed the cap,
an `other` bucket is retained.

Encoding now reserves row width up-front
to reduce repeated reallocation overhead.

---

## 18) Explainability Modes

Supported explainability modes:

- `permutation`,
- `integrated_gradients`,
- `hybrid`.

Hybrid blend weights are configurable:

- `hybrid_explainability_weight_permutation`,
- `hybrid_explainability_weight_integrated_gradients`.

Weights are validated and normalized at runtime.

---

## 19) Bivariate Selection Logic

Bivariate significance selection combines:

- statistical significance,
- correlation strength,
- neural relevance,
- bounded Tier-3 fallback promotion when neural yield is sparse.

Supported strategy values:

- `auto`,
- `balanced`,
- `corr_heavy`,
- `importance_heavy`.

For large datasets,
fast mode can cap pairwise evaluation counts.

Tier-3 aggressiveness is tunable using:

- `bivariate_tier3_fallback_aggressiveness` (0..3).

---

## 20) Output Directory Behavior

Default output path:

- `<dataset_stem>_seldon_outputs`

Inside output directory:

- markdown reports,
- plot assets folder,
- optional HTML report files.

You can override output root using:

- `--output-dir <path>`

---

## 21) Preprocessed Export

Export format control:

- `export_preprocessed=none|csv|parquet`

Path base control:

- `export_preprocessed_path`.

CLI examples:

```bash
./seldon /path/data.csv --export-preprocessed csv
./seldon /path/data.csv --export-preprocessed parquet --export-preprocessed-path /tmp/prep_sales
```

---

## 22) Plotting

Plot toggles:

- `plot_univariate`,
- `plot_overall`,
- `plot_bivariate_significant`,
- `plots` alias list.

Plot rendering settings:

- `plot_format`,
- `plot_theme`,
- `plot_grid`,
- `plot_width`,
- `plot_height`,
- `plot_point_size`,
- `plot_line_width`.

---

## 23) Logging Behavior

Seldon supports both verbose logs and progress spinner output.

Current behavior:

- spinner is active for non-verbose TTY runs,
- verbose mode emits line-based logs without spinner clashes.

This prevents garbled mixed output in terminal sessions.

---

## 24) Example: Clean Fast Run

```bash
./seldon /path/data.csv \
  --profile quick \
  --plots none \
  --neural-strategy auto \
  --bivariate-strategy auto
```

---

## 25) Example: High-Control Analytical Run

```bash
./seldon /path/data.csv \
  --target revenue \
  --profile thorough \
  --feature-strategy adaptive \
  --neural-strategy balanced \
  --bivariate-strategy balanced \
  --neural-max-one-hot-per-column 32 \
  --neural-max-topology-nodes 4096 \
  --neural-max-trainable-params 20000000 \
  --feature-engineering-max-generated-columns 700 \
  --export-preprocessed csv
```

---

## 26) Example: Schema-Guided Run

```bash
./seldon /path/data.csv \
  --type invoice_date:datetime \
  --type region:categorical \
  --type amount:numeric \
  --target amount \
  --plots bivariate,overall
```

---

## 27) Example Config File

```yaml
dataset: /absolute/path/to/data.xlsx
target: churn_score
profile: thorough
output_dir: /tmp/seldon_churn

datetime_locale_hint: auto
numeric_locale_hint: auto

type.signup_date: datetime
type.plan_tier: categorical
type.monthly_spend: numeric

impute.monthly_spend: median
impute.plan_tier: mode

outlier_method: iqr
outlier_action: cap
store_outlier_flags_in_report: false

scaling: auto

feature_engineering_enable_poly: true
feature_engineering_enable_log: true
feature_engineering_degree: 3
feature_engineering_max_base: 10
feature_engineering_max_generated_columns: 640

neural_strategy: auto
neural_optimizer: lookahead
neural_lookahead_fast_optimizer: adam
neural_lookahead_sync_period: 5
neural_lookahead_alpha: 0.5
neural_learning_rate: 0.001
gradient_clip_norm: 5.0

neural_max_one_hot_per_column: 32
neural_max_topology_nodes: 4096
neural_max_trainable_params: 20000000

neural_explainability: hybrid
hybrid_explainability_weight_permutation: 0.5
hybrid_explainability_weight_integrated_gradients: 0.5
neural_uncertainty_samples: 24
neural_ensemble_members: 3
neural_ensemble_probe_rows: 256
neural_ensemble_probe_epochs: 48
neural_ood_enabled: true
neural_ood_z_threshold: 3.5
neural_ood_distance_threshold: 2.5
neural_drift_psi_warning: 0.15
neural_drift_psi_critical: 0.25

plots: all
plot_format: png
plot_theme: auto
plot_grid: true
generate_html: false
```

---

## 28) Frequent Operational Checks

Check executable:

```bash
./seldon --interactive
```

Check converter tools:

```bash
command -v xlsx2csv
command -v xls2csv
command -v gzip
command -v unzip
```

Check plotting/HTML tools:

```bash
command -v gnuplot
command -v pandoc
```

---

## 29) Troubleshooting Quick Notes

If execution is memory-heavy:

- lower `feature_engineering_max_generated_columns`,
- lower `neural_max_one_hot_per_column`,
- use `profile quick`,
- reduce `neural_importance_max_rows`.

If type inference is not ideal:

- apply `--type` overrides,
- or set `type.<column>` keys in config.

If runs are too slow:

- set `fast_mode=true`,
- lower `fast_max_bivariate_pairs`,
- lower `fast_neural_sample_rows`,
- reduce explainability sampling settings.

---

## 30) Documentation Map

Detailed runtime usage:

- [docs/USAGE.md](docs/USAGE.md)

Internal architecture and module flow:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 31) License

Seldon is distributed under the MIT License.

See:

- [LICENSE](LICENSE)
