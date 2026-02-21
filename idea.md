# Seldon: Agentic Data Analytics Engine

## Overview
Seldon is an experimental, agent-driven Command Line Interface (CLI) built in C++ using AntiGravity. It acts as a reliable data science consultant, designed to automate the repetitive boilerplate of statistical analysis so the user can focus on high-level inference and ideas. 

Rather than overwhelming the user with massive analytics dumps, Seldon processes datasets in logical, permission-based layers. It calculates, assesses, and only presents the most statistically significant findings, pausing at major checkpoints to ask the user for permission to proceed deeper.

## Core Philosophy
1. **Layered Intelligence:** Move from simple (univariate) to complex (multivariate) only when mathematically justified.
2. **Cognitive Protection:** Never bombard the user. Filter out statistical noise and present curated, high-value insights.
3. **Agentic Autonomy (with Permission):** The system decides *what* tests to run based on the data's shape, but the human decides *if* it should proceed.

---

## System Architecture: The Three Layers



### Layer 1: The Foundation (Univariate Analysis)
**Trigger:** User ingests a multi-variate dataset (CSV).
**Action:** Seldon automatically parses the file, isolates numerical columns, and performs foundational statistical sweeps.
**Output:** Presents concise summary tables.
* **Metrics Tracked:** * Central Tendencies (Mean, Median)
  * Dispersion (Variance, Standard Deviation)
  * Shape (Skewness, Kurtosis)
* **Agentic Checkpoint:** Seldon pauses and asks: *"Foundation analysis complete. Proceed to Bivariate pattern hunting? (Y/n)"*

### Layer 2: The Inference (Bivariate & Predictive Routing)
**Trigger:** User grants permission.
**Action:** Seldon calculates a correlation matrix for all combinations of rows/columns. It acts autonomously here, filtering out low correlations to protect the user from noise.
**Agentic Routing:**
* **High Correlation (2 Variables):** Automatically performs Simple Linear Regression.
* **High Correlation (Multiple Variables):** Automatically escalates to Multiple Linear Regression.
* **High Predictive Viability:** If regression yields strong positive results (e.g., high R-squared), Seldon preps the data for rigorous inference by building a foundational Neural Network model for future predictions.
**Output:** Presents only the *n* most significant findings (the software decides the threshold based on significance levels).
* **Agentic Checkpoint:** Seldon pauses and asks: *"Inference mapped. Do you want to initialize a combined, deep multivariate analysis? (Y/n)"*

### Layer 3: The Synthesis (Combined Multivariate Analysis)
**Trigger:** User explicitly opts-in.
**Action:** Performs complex, combined interactions on the dataset based on the pathways discovered in Layer 2. This is the heavy-lifting phase for rigorous statistical inference, custom grouping, or advanced predictive modeling.
**Output:** Final synthesized report and model readiness for coursework/presentations.

---

## Development Roadmap (C++ & AntiGravity)
* **Phase 1: Ingestion & Layer 1:** Build the CSV parser and standard math functions (Mean, StdDev, Skewness). Format terminal outputs into clean tables.
* **Phase 2: The Logic Engine:** Implement the correlation matrix calculations and the `if/else` decision trees that allow Seldon to decide between simple and multiple regression.
* **Phase 3: The Agentic Loop:** Build the CLI prompt system. Ensure state is preserved between Layer 1, 2, and 3 so the user can easily navigate the pipeline.
* **Phase 4: Neural Net Integration:** Implement basic feed-forward network logic for the final predictive layer.