# Stochastic Gradient-Guided Estimation of Local Lipschitz Constants in Neural Networks

This repository contains the official implementation and experimental pipeline for the paper:
> **Stochastic Gradient-Guided Estimation of Local Lipschitz Constants in Neural Networks**  
> Xavier Sierra-Canto, Ricardo Legarda-Saenz, Carlos Brito-Loeza  

---

## Abstract
Estimating the local Lipschitz constant of a neural network is central to understanding its robustness, but exact computation is NP-hard. We revisit MonteLip, a simulated-annealing-based lower-bound estimator, and identify a failure mode of its isotropic search: the expected alignment between a random direction and the dominant sensitivity vector decays as $O(1/\sqrt{n})$. We propose **MonteLip-G**, replacing isotropic proposals with gradient-guided updates to maintain high recovery across high input dimensions and network depths.

---

## Repository Structure
- **`run_experiment.py`**: Runs the main dimension and depth sweeps against reference baselines.
- **`ablation_sa_components.py`**: Evaluates the factorial ablation of simulated annealing components (noise, Metropolis criterion, and cooling schedule).
- **`clever.py` & `lipbab_wrapper.py`**: External baseline integrations.

---

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/your-username/MonteLip2026.git](https://github.com/your-username/MonteLip2026.git)
cd MonteLip2026
pip install -r requirements.txt
