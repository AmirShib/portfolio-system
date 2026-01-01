# Resilient Portfolio System (RPS)


**A Quantitative Investment Platform for experimenting with different portfolio optimizations.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The **Resilient Portfolio System (RPS)** is an end-to-end quantitative research platform designed to construct, analyze, and stress-test investment portfolios using advanced machine learning and statistical techniques.

Unlike simple "stock screener" apps, RPS implements institutional-grade algorithms.Including **L1 Regularized Optimization (Lasso)**, **Nested Clustered Optimization (NCO)**, and **Regime-Switching Monte Carlo Simulations** to build portfolios that are mathematically robust to market crashes (hopefully...).

---

## Key Features 

### Math Engine (work in progress)
* **L1 Regularization (Sparse Portfolios):** Uses Convex Optimization (`cvxpy`) to mathematically force zero-weight to redundant or risky assets, creating cleaner, more explainable portfolios.
* **NCO-RMT (Denoising):** Applies **Random Matrix Theory** to filter statistical noise from the covariance matrix and uses **Hierarchical Clustering** to allocate risk across uncorrelated clusters (e.g., Tech vs. Energy) rather than individual stocks.
* **Regime-Switching Simulation:** Uses **Hidden Markov Models (HMM)** to detect "Bull" vs. "Bear" market regimes and runs 1,000+ path-dependent Monte Carlo simulations to predict drawdown risk.

### High-Performance Architecture
* **Microservices-Ready:** Backend (API) and Frontend (UI) are decoupled and containerized.
* **JIT Compilation:** Critical simulation loops are compiled to machine code using **Numba**, which can run over 50x faster than standard Python.
* **Vectorized Backtesting:** Uses **VectorBT** to run 5-year strategy backtests (SMA, RSI, etc.).
* **Smart Caching:** Implements a **Redis** Cache-Aside pattern to prevent redundant recalculations.

### Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Core Logic** | Python 3.11, NumPy, Pandas | The brain of the system. |
| **Optimization** | CVXPY, Scikit-Learn | Solving convex problems & clustering. |
| **Simulation** | Numba, HMMlearn | High-speed stochastic modeling. |
| **Backend** | FastAPI, Uvicorn | Async API server. |
| **Frontend** | Streamlit, Plotly | Interactive research dashboard. |
| **Data & Cache** | Yahoo Finance, Redis | Market data and state management. |
| **DevOps** | Docker, Docker Compose | Containerization and orchestration. |

---

## Architecture

The system follows a modular design seeking scalability and separation of concerns:

```mermaid
graph TD
    User[User / Analyst] -->|HTTP| Streamlit[Streamlit Frontend]
    Streamlit -->|JSON| API[FastAPI Backend]
    
    subgraph "Engine Room (Dockerized)"
        API -->|Read/Write| Redis[(Redis Cache)]
        API -->|Fetch| YF[Yahoo Finance API]
        
        API -->|Dispatch| Core[Core Logic Module]
        Core -->|Solve| Opt[L1 / NCO Optimizer]
        Core -->|Simulate| Sim[Numba Monte Carlo]
        Core -->|Backtest| VBT[VectorBT Engine]
    end