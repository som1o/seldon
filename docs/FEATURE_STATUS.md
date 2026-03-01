# Seldon Feature Status (March 2026)

This file flags areas that are still maturing and records what has been hardened in the latest pass.

## Hardened in this pass

- **HTML export status accuracy**
  - Previous behavior: when `generate_html=true` and `pandoc` was unavailable (or conversions failed), output looked successful.
  - Current behavior: pipeline now reports explicit warnings and prints conversion success counts.

- **Parquet export resilience**
  - Previous behavior: parquet conversion used a Python bridge and silent best-effort behavior.
  - Current behavior: optional native C++ parquet writer path (Arrow/Parquet) is used; Python bridge removed from runtime export flow.

- **Runtime benchmark mode**
  - Current behavior: optional benchmark mode records per-stage wall-clock timings and pair-analysis throughput in generated reports.

- **Readiness matrix in reports**
  - Current behavior: generated reports now include a subsystem readiness matrix (`stable`, `experimental`, `best-effort`) with guardrails.

- **Bivariate pair-analysis hot path**
  - Previous behavior: pair loop repeated expensive lookups/string transforms and structural string-key assembly.
  - Current behavior: precomputes per-feature metadata, uses numeric structural keys, and removes repeated per-pair token normalization.

- **Deployable REST prediction service**
  - Previous behavior: inference was tied to offline pipeline execution.
  - Current behavior: `seldon --serve` loads persisted neural binaries via model registry, serves `/predict` and `/batch_predict`, supports thread-pooled concurrency, and logs live request/latency/distribution monitoring.

## Ongoing guardrails (intentionally explicit)

- **GPU acceleration paths (`OpenCL`/`CUDA`)**
  - Build toggles are available and degrade gracefully, but acceleration effectiveness remains environment-dependent and should be validated per hardware stack.

- **Causal orientation outputs**
  - Causal graph directionality is now produced via robust algorithmic consensus (PC/Meek + GES + LiNGAM + bootstrap + temporal proxy validation), and should still be interpreted as decision support rather than intervention-grade proof.

- **Native parquet dependency chain**
  - Native parquet export requires Arrow/Parquet C++ libraries at build time; binaries without these dependencies emit a clear runtime warning and keep CSV export.

## Completed roadmap items

1. Optional native parquet writer path (pure C++ dependency) to remove Python bridge.
2. Benchmark mode with per-stage wall-clock timing and pair-analysis throughput.
3. Structured readiness matrix in generated reports (`stable`, `experimental`, `best-effort`).
