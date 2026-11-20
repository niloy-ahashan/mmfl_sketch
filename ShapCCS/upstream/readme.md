# ShapCCS: Client Selection via Shapley Value Estimation in Federated Learning

This repository contains the implementation of **ShapCCS**, a client-level coreset selection framework for federated learning.

## Abstract

Computation overhead has become a critical bottleneck for Federated Learning (FL). Coreset selection solves this issue by constructing a compact and informative subset to represent the full dataset. However, existing approaches optimize coreset construction only at the data level and enforce an identical retention ratio among all clients, ignoring client heterogeneity and introducing detrimental fragmented clients. In this paper, we first introduce a gradient projection Shapley value (GPSV) to measure client contributions. GPSV captures both the direction and magnitude information of client updates and enables exact Shapley value calculation with $\mathcal{O}(1)$ per-coalition evaluation. Based on GPSV, we then propose ShapCCS, the first client-level coreset selection strategy for FL that prioritizes clients with high GPSV and excludes fragmented clients with negligible or even negative scores. As a client-level coreset selection strategy, ShapCCS can be combined with a data-level selection strategy, and additionally saves communication costs, an advantage that cannot be achieved by the data-level approaches. Extensive experiments demonstrate the superiority of ShapCCS on model performance and robustness to noise.

## Dependencies

This project requires the following Python packages:

- PyTorch (`torch`)
- scikit-learn(`sklearn`)
- tqdm(`tqdm`)

## How to Run Experiments

All experiments are defined and launched from a single entry point: `run_prune.py`. You do not need to pass command-line flags—simply edit the configuration variables in the script and run it.

Open the `run_prune.py` and modify the following configuration blocks:

- **`seeds`**: Random seeds for reproducibility.
- **`prune_fracs`**: Fraction of pruing data.
- **`num_clients_configs`**: Number of clients.
- **`dirichlet_alphas`**: Dirichlet parameters.
- **`ml_methods`**: Data-level corset selection methods for ShapCCS.
- **`num_clients_remain_config`**: Expected number of clients remained after selection. 

Then run the script:

```
bash
python run_prune.py
```

