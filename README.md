# Seldon: Agentic Data Analytics Engine

Seldon is an experimental, agent-driven Command Line Interface (CLI) built natively in C++17. Designed as a reliable, zero-dependency data science consultant, Seldon automates the repetitive boilerplate of statistical analysis so you can focus on high-level inference and ideas.

Rather than overwhelming you with massive unstructured analytics dumps, Seldon processes datasets in logical, permission-based layers. It calculates, evaluates, filters noise, and only presents the most statistically significant findings, pausing at major checkpoints to ask for your authorization to proceed deeper.

## Core Philosophy

1. **Layered Intelligence:** Move from simple (univariate) to complex (multivariate) only when mathematically justified.
2. **Cognitive Protection:** Never bombard the user. Filter out statistical noise and present curated, high-value insights.
3. **Agentic Autonomy (with Permission):** The system decides *what* tests to run based on the data's topological shape, but the human specifically decides *if* it should proceed.

---

## Architectural Highlights

### 1. Resilient Columnar Memory Ingestion
Data consistency in the real world is messy. Seldon's internal CSV parser (`Dataset::load()`) is a reliable consultant. It maps datasets directly into high-locality columnar formats to optimize algorithmic access. It includes a strict parsing policy to purposefully reject and skip malformed or missing rows instead of injecting statistical bias with zeros. 

### 2. The Agentic Threshold (`r = 0.6`)
In Phase 2 (The Logic Engine), Seldon automatically computes the Pearson Correlation matrices across the entire dataset. However, Seldon makes a strict, opinionated design choice to default the `agenticThreshold` to `0.6`. This mechanism explicitly prevents information overload. It filters out weak signals to protect the user's cognitive load, guaranteeing that Seldon only brings forward the most mathematically "meaningful" stories.

### 3. P-Value Filtering
High correlation is meaningless without statistical significance. The Logic Engine evaluates the precise Student's t-statistic for every correlation via exact finite series integration, autonomously dropping variables mathematically if the two-tailed p-value exceeds `0.05`.

### 4. Robust Householder QR Decomposition
To protect the system from the Singular Matrix trap (which plagues naïve Inverse Normal equations), Seldon performs Multiple Linear Regression seamlessly via Householder QR Decomposition. It natively absorbs perfectly collinear data logic via back-substitution, allowing the Logic Engine to smoothly model multi-variable clusters.

### 5. Dynamic Neural Topology & Synthesis
Phase 3 (Synthesis) engages a custom C++ Feed-Forward Neural Network trained via Stochastic Gradient Descent backpropagation. Rather than predicting randomness or hardcoding the network architecture, Seldon evaluates the regression clusters to explicitly construct matching `[Input -> Hidden -> Output]` topologies, mapping the relationships found into real predictive neural weights before validating inferences.

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
* **Action:** Deploys and explicitly trains a dynamically modeled Neural Network.
* **Logic:** Normalizes the highest-correlated regression combinations via Min-Max scalars, initiating Stochastic Gradient Descent over 100 epochs to establish functional feature predictions. To prevent hardware saturation on massive datasets, the neural loop intrinsically throttles execution to strictly `80% CPU utilization` maximum per core.
* **Agentic UI:** Displays actual MSE loss depreciation during learning phases before dropping its final inference map.

---

## Building and Running

Seldon requires zero external libraries. It requires C++17 and CMake.

```bash
mkdir build && cd build
cmake ..
make

# Run the agent over your dataset
./seldon .path/name_of_dataset.csv
```
