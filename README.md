# Project Plan: A Deep-Learning "Fragility Index" for Financial Markets

**Project Objective:** To implement the deep-learning framework from Liu et al. (2024) [cite_start][cite: 3] to predict critical transitions in financial networks. We will first generate a massive synthetic dataset of financial markets, each with a known "brittleness" tipping point. We will then train a GIN-GRU model to predict this point from pre-transition time-series data. Finally, we will apply this trained model to real S&P 500 data to generate a novel "Systemic Fragility Index" and test its ability to provide early warnings for the 2008 financial crisis.

---

## Phase 1: Synthetic Data Generation (The "Training Universe")

**Goal:** Create a massive and diverse dataset of synthetic financial markets, each with a precisely known critical transition "label" ($\epsilon_c$).

### 1.1. The Synthetic System
Our "toy" universe will consist of simulated stock markets.
* **Nodes:** `N` stocks, where `N` is randomly chosen for each simulation (e.g., between 50 and 200).
* **Structure:** The `N` stocks are grouped into `K` random clusters (e.g., "Tech," "Finance," "Health").
* **Control Parameter ($\epsilon$):** We define $\epsilon$ as the **"inter-cluster correlation."** This is our "fragility" knob.
    * When $\epsilon$ is low (e.g., 0.1), the market is healthy and diversified. "Tech" and "Finance" stocks move independently.
    * When $\epsilon$ is high (e.g., 0.9), the market is brittle and fragile. All stocks move together.

### 1.2. The Simulation & Labeling Process
For each *single* simulation in our dataset:
1.  **Generate Data:** We loop $\epsilon$ from 0.1 to 0.9. At each step, we generate `T=100` time steps of synthetic stock *returns* for all `N` stocks using a multivariate normal distribution defined by our correlation matrix.
2.  **Find the "Tipping Point" ($\epsilon_c$):** After the simulation, we analyze the data to find the *true* critical transition. We define this as the **value of $\epsilon$ where the largest eigenvalue ($\lambda_{max}$) of the correlation matrix begins to spike.** This $\lambda_{max}$ represents the "market mode" (how much the entire market moves as one), and its sudden growth is a classic sign of a critical transition. [cite_start]This $\epsilon_c$ is our quantitative **label**[cite: 56].
3.  **Create Training Samples:** We go back to the time-series data generated *before* the system reached $\epsilon_c$. [cite_start]We use a **sliding window** (e.g., `w=20` time steps, as in the paper [cite: 61, 203]) to create many `(N_stocks, w_timesteps)` samples.
4.  [cite_start]**Assign Label:** *Every single sample* generated from this one simulation gets the **exact same label**: the $\epsilon_c$ we found in Step 2[cite: 81].
5.  [cite_start]**Repeat:** We run this entire process 1,000+ times, each time with a different `N`, `K`, and resulting $\epsilon_c$, to build a "massive and diverse dataset"[cite: 91].

---

## Phase 2: Model Architecture & Training

[cite_start]**Goal:** Implement the GIN-GRU architecture [cite: 43] to learn the relationship between a pre-transition time series (`X_s`) and its ultimate tipping point (`\epsilon_c`).

### 2.1. The GIN-GRU Architecture
Our model will be a regressor that outputs a single number.
1.  **Input:** A single training sample `X_s` with shape `(N, w)`. `N` is the number of stocks, `w` is the window length (e.g., 20).
2.  **GIN (Graph Isomorphism Network) Layers:** At each time step `t` (from 1 to `w`), the GIN layers process the *spatial* snapshot of `N` stock returns. [cite_start]This part acts as a "graph encoder" that learns the collective state of the system *at that instant*[cite: 44].
3.  [cite_start]**Global Max Pooling (GMPool):** After the GIN, a pooling operation "flattens" the `N` node representations into a single graph-level embedding `h_t` for that time step[cite: 84].
4.  **GRU (Gated Recurrent Unit) Layer:** The *sequence* of graph embeddings `[h_1, h_2, ..., h_w]` is fed into the GRU. [cite_start]This part of the model reads the *temporal* patterns, learning how the network's collective state evolves as it approaches the transition[cite: 45].
5.  [cite_start]**MLP (Multilayer Perceptron) Head:** The final output vector from the GRU is passed to a simple MLP that regresses it down to a **single output number**: our predicted tipping point, $\tilde{\epsilon}_c$[cite: 85, 107].

### 2.2. Training
* **Task:** Train the model on the thousands of `(X_s, \epsilon_c)` pairs from Phase 1.
* [cite_start]**Loss Function:** We use the **Mean Square Error (MSE)**[cite: 108]. [cite_start]The model is directly penalized for the squared difference between its prediction $\tilde{\epsilon}_c$ and the true label $\epsilon_c$[cite: 109].

---

## Phase 3: Real-World Application & Analysis

**Goal:** Use our *trained* model as a "measurement instrument" to see if it can detect the build-up of risk for the 2008 financial crisis.

### 3.1. Data Sourcing & Preparation
1.  **Stock Data:** We will download daily "Adjusted Close" prices for all S&P 500 companies from **2005-2010** using the `yfinance` Python library. This data is publicly and freely available.
2.  **Validation Data:** We will also download the daily prices for the **CBOE Volatility Index (VIX)** (ticker: `^VIX`) from the same source. The VIX is our benchmark for *contemporaneous* market panic.

### 3.2. Generating the "Systemic Fragility Index" (SFI)
This is the core test of our project.
1.  **Create Input Stream:** We create a **rolling window** over our real S&P 500 data. On any given day `D` (e.g., June 1, 2007), we take the last `w=20` days of stock *returns* for all 500 companies. This creates a real-world tensor of shape `(500, 20)`.
2.  **Make Prediction:** We feed this *single* real-world tensor into our *trained model* from Phase 2. The model (which has *only* ever seen synthetic data) will output a single number, $\tilde{\epsilon}_c$.
3.  **Plot the Index:** We record this output number for day `D`. We then slide our window forward one day to `D+1` and repeat the process. The resulting time series of $\tilde{\epsilon}_c$ values is our **"Systemic Fragility Index."**

### 3.3. Analysis & Conclusion
Our final deliverable will be a chart comparing our index to the VIX.
* **The "A-ha!" Plot:** We will plot our SFI against the VIX from 2005-2010.
* **Hypothesis:** The VIX is known to spike *during* a crisis (e.g., late 2008). Our SFI is designed to be an *early warning signal*.
* **Success Criterion:** The project is a success if our SFI shows a clear, sustained, and significant upward trend starting in **2007**, demonstrating that the model "saw" the market's underlying fragility building up *well before* the VIX spiked and the market collapsed.