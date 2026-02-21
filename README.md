# Seldon: The Agentic Data Analytics Engine

> *"In the flood of data, the human requires a lighthouse, not more water."*

Seldon is a high-performance, agent-driven Command Line Interface (CLI) analytics engine built natively in C++17. It is designed as a **Reliable Consultant**—a zero-dependency mathematical powerhouse that automates statistical analysis, providing high-precision diagnostics and neural synthesis with extreme efficiency.

---

## 🚀 Modernization Highlights (Phases 1-5)

The Seldon Engine has been completely modernized to handle massive datasets and professional workflows:

- **Extreme Scalability**: Core engines (Stats, Logic, Neural) are now multi-threaded via **OpenMP** and utilize **SIMD vectorization** for high-throughput arithmetic.
- **Mathematical Rigor**: Replaced numerical integration with **Lentz's method for Continued Fractions** in `betainc`, providing high-precision p-values for extreme correlations and sample sizes.
- **Deep Regression Diagnostics**:
  - **Simple Linear**: Standard Errors, t-stats, p-values, and 95% Confidence Intervals.
  - **Multiple Linear**: Adjusted R², F-statistic (with significance testing), and collinearity protection.
- **Advanced Neural Features**: Step-decay **Learning Rate Scheduling**, Binary **Cross-Entropy** loss, and early stopping with plateau detection.
- **Explainable AI (XAI)**: Native **Permutation Feature Importance** to identify which inputs most significantly drive model accuracy.
- **High-Speed Persistence**: Optimized **Binary Model Serialization** (`.seldon`) for near-instant save/load cycles.
- **Modern CLI**: Structured configuration system with short-flags and automated results export to JSON/CSV.

---

## 🧭 The Seldon Workflow: A Typical Session

1.  **Ingestion & Scouring**: Seldon scours your CSV for numerical density, identifying features with real analytical signal. Real-time progress indicators track ingestion status.
2.  **The Foundation Summary**: View Skewness, Kurtosis, and basic moments in a dynamically formatted, adaptive terminal table.
3.  **The Bivariate Pivot**: Seldon identifies statistically significant relationships (|r| > threshold, p < 0.05). Findings are presented with full diagnostic payloads.
4.  **The Predictive Synthesis**: Authorization triggers the neural lattice training. Watch convergence in real-time as the engine detects whether to use MSE for regression or Cross-Entropy for classification.
5.  **Explainability Report**: Post-training, Seldon disruptively shuffles features to calculate their relative importance, giving you a deep look into the "Black Box."

---

## 🏛️ Architectural Blueprint: Stability Over Speed

### Numerical Integrity & QR Factorization
In Multiple Linear Regression, we utilize **Householder QR Decomposition**. By factorizing the matrix into $Q$ (orthogonal) and $R$ (triangular), Seldon avoids the instability of direct matrix inversion.

### Analytical Precision
Significance is computed via the regularized incomplete beta function. We avoid series expansion approximations in favor of continued fraction evaluation, ensuring stable p-values even for datasets with 10M+ records.

---

## 🛠️ Configuring the Agent (CLI)

| Argument | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--threshold-bivariate` | `-tb` | Correlation threshold for simple regression | `0.6` |
| `--threshold-mlr` | `-tm` | Threshold for MLR inclusion | `0.35` |
| `--exhaustive-scan` | | Scan all rows to identify numeric columns | `false` |
| `--epochs` | | Neural network training epochs | `300` |
| `--lr` | | Neural network learning rate | `0.02` |
| `--impute` | | Missing value handling (`skip`, `zero`, `mean`, `median`) | `zero` |
| `--output` | `-o` | Export findings to JSON/CSV | |
| `--batch` | | Non-interactive mode (auto-accept prompts) | `false` |
| `--verbose` | | Enable detailed epoch-by-epoch logs | `false` |

---

## 📝 Model Persistence & Export

Seldon supports high-performance binary storage and structured result exports for integration with Python/R dashboards.

```bash
# Analyze a dataset and export findings to JSON
./seldon data.csv --batch --output results.json

# Train a model and save the binary weights
./seldon data.csv --epochs 500 --batch
```

---

## 🏗️ Building Seldon

Seldon requires a C++17 compiler and OpenMP support.

```bash
mkdir build && cd build
cmake ..
make
```

---

## ❤️ Credits and License

Seldon is built as a study of agentic systems and high-performance C++. 
Distributed under the MIT License. See `LICENSE` for more information.

> *Seldon is named after Hari Seldon, the father of Psychohistory—the mathematical study of patterns in vast systems.*
