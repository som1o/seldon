# Seldon: Agentic Data Analytics Engine

Seldon is an experimental, agent-driven Command Line Interface (CLI) built natively in C++17. Designed as a reliable, zero-dependency data science consultant, Seldon automates the repetitive boilerplate of statistical analysis so you can focus on high-level inference and ideas.

Rather than overwhelming you with massive unstructured analytics dumps, Seldon processes datasets in logical, permission-based layers. It calculates, evaluates, filters noise, and only presents the most statistically significant findings, pausing at major checkpoints to ask for your authorization to proceed deeper.

## Core Philosophy

1. **Layered Intelligence:** Move from simple (univariate) to complex (multivariate) only when mathematically justified.
2. **Cognitive Protection:** Never bombard the user. Filter out statistical noise and present curated, high-value insights.
3. **Agentic Autonomy (with Permission):** The system decides *what* tests to run based on the data's topological shape, but the human specifically decides *if* it should proceed.

---

## Architectural Highlights

### 1. Resilient Memory Ingestion
Data consistency in the real world is messy. Seldon's internal CSV parser (`Dataset::load()`) acts as a reliable consultant. It reads files exceptionally quickly while utilizing a `try/catch` block during column vectorization to gracefully catch and handle non-numerical data mapping errors without crashing. Bad data is silenced, and the flow continues. 

### 2. The Agentic Threshold (`r = 0.6`)
In Phase 2 (The Logic Engine), Seldon automatically computes the Pearson Correlation matrices across the entire dataset. However, Seldon makes a strict, opinionated design choice to default the `agenticThreshold` to `0.6`. This mechanism explicitly prevents information overload. It filters out weak signals to protect the user's cognitive load, guaranteeing that Seldon only brings forward the most mathematically "meaningful" stories.

### 3. P-Value Filtering
High correlation is meaningless without statistical significance. The Logic Engine evaluates the Student's t-statistic for every correlation and drops variables passing the threshold if their approximate p-value exceeds `0.05`.

### 4. The Singular Matrix Trap
When variables are perfectly collinear (a multiple of another), the Normal Equation design matrix (`X^T * X`) becomes singular and non-invertible. Rather than throwing a fatal crash during Gaussian Elimination, Seldon's mathematical layer quietly returns an empty matrix, allowing the Logic Engine to elegantly abort the calculation with a `[Agent Protection]` notice, ensuring total application resilience.

### 5. Dynamic Neural Topology
Phase 3 (Synthesis) engages a custom C++ Feed-Forward Neural Network. Rather than hardcoding the network architecture, Seldon evaluates the clustering complexity found during the Multiple Linear Regression stage and dynamically scales the Neural Net's `[Input -> Hidden -> Output]` layer counts to match the data topological complexity.

---

## Agentic Permission Flow

Seldon fundamentally respects the human operator by halting execution at major analytical boundaries using `std::getline` based Y/n prompts. It maps the progression perfectly without suffering from `std::cin` buffer leakage, ensuring the operator genuinely authorizes the dive into deeper mathematical modeling.

## The Three Agentic Layers

### Layer 1: The Foundation (Univariate)
* **Action:** Automatically parses datasets, isolating and validating numerical columns.
* **Math:** Computes standard statistics: Mean, Median, Variance, StdDev, Skewness, Kurtosis.
* **Agentic UI:** Displays a perfectly formatted Foundation Summary table, pausing to ask: *"Proceed to Bivariate pattern hunting? (Y/n)"*

### Layer 2: The Inference (Bivariate & Predictive Routing)
* **Action:** Constructs Correlation Matrix and evaluates significance. 
* **Logic:** Automatically filters noise using the Agentic Threshold and P-Value checks. Triggers OLS Simple Linear Regressions for pairs, and Normal Equation MLR for multi-variable clusters.
* **Agentic UI:** Reports the specific linear equations and relationships found, then asks: *"Inference mapped. Do you want to initialize a combined, deep multivariate analysis? (Y/n)"*

### Layer 3: The Synthesis (Predictive Neural Network)
* **Action:** Deploys a dynamically topologically scaled Neural Network.
* **Logic:** Analyzes data nodes through hidden activation layers to map deep feature synthesis predictions.
* **Agentic UI:** Dumps finalized simulated inference model state signaling readiness for the user's coursework or presentation.

---

## Building and Running

Seldon requires zero external libraries. It requires C++17 and CMake.

```bash
# Compile via typical CMake or GCC
g++ -std=c++17 -I include/ src/*.cpp -o seldon

# Run the agent over your dataset
./seldon data.csv
```
