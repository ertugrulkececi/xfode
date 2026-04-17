# xFODE: An Explainable Fuzzy Additive ODE Framework for System Identification

Official MATLAB implementation of the paper:

> **xFODE: An Explainable Fuzzy Additive ODE Framework for System Identification**
> Ertuğrul Keçeci, Tufan Kumbasar
> IEEE Conference on Artificial Intelligence (CAI), 2026
> [arXiv:2604.14883](https://arxiv.org/abs/2604.14883)

---

## Overview

Recent advances in Deep Learning (DL) have strengthened data-driven System Identification (SysID), with Neural and Fuzzy Ordinary Differential Equation (NODE/FODE) models achieving high accuracy in nonlinear dynamic modeling. Yet, system states in these frameworks are often reconstructed without clear physical meaning, and input contributions to the state derivatives remain difficult to interpret. To address these limitations, we propose Explainable FODE (xFODE), an interpretable SysID framework with integrated DL-based training. In xFODE, we define states in an incremental form to provide them with physical meanings. We employ fuzzy additive models to approximate the state derivative, thereby enhancing interpretability per input. To provide further interpretability, Partitioning Strategies (PSs) are developed, enabling the training of fuzzy additive models with explainability. By structuring the antecedent space during training so that only two consecutive rules are activated for any given input, PSs not only yield lower complexity for local inference but also enhance the interpretability of the antecedent space. To train xFODE, we present a DL framework with parameterized membership function learning that supports end-to-end optimization. Across benchmark SysID datasets, xFODE matches the accuracy of NODE, FODE, and NLARX models while providing interpretable insights.

---

## Requirements

- MATLAB R2023b 
- Deep Learning Toolbox
- System Identification Toolbox

---

## Repository Structure

```
xfode/
├── xFODE/              # Proposed method (xFODE-PS1/PS2/PS3 and AFODE)
│   ├── run.m           # Entry point — configure dataset, PS, and SR here
│   └── lib/            # Core library functions
├── FODE/               # FODE baseline
│   ├── run.m
│   └── lib/
├── NODE/               # NODE baseline
│   ├── run.m
│   └── lib/
└── datasets/
    └── EVBattery/      # EV Battery dataset 
```

---

## Quick Start

1. Open MATLAB and navigate to the model folder (e.g., `xFODE/`).
2. Open `run.m` and set your configuration:
   ```matlab
   dataset_name          = "MRDamper";     % TwoTank | HairDryer | MRDamper | EVBattery | SteamEng
   SR_method             = "incremental"; % "lagged" (SR1) | "incremental" (SR2)
   input_membership_type = "trimf";       % "gaussmf" (AFODE) | "trimf" (PS1) | "gauss2mf" (PS2) | "c-gauss2mf" (PS3)
   number_of_runs        = 20;
   ```
3. Run `run.m`. Results (mean ± std of RMSE over `number_of_runs` seeds) are printed to the console.

---

## Datasets

| Dataset | Inputs | Outputs | Train samples | Test samples | Source |
|---------|--------|---------|---------------|--------------|--------|
| Two-Tank | 1 | 1 | 1500 | 1500 | Built-in (`twotankdata`) |
| Hair Dryer | 1 | 1 | 500 | 500 | Built-in (`dryer2`) |
| MR Damper | 1 | 1 | 3000 | 499 | Built-in (`mrdamper`) |
| Steam Engine | 2 | 2 | 250 | 201 | Built-in (`SteamEng`) |
| EV Battery | 2 | 1 | 15001 | 14351 | `datasets/EVBattery/` |

The Hair Dryer, MR Damper, Two-Tank, and Steam Engine datasets ship with MATLAB's System Identification Toolbox and are loaded by name. Only the EV Battery dataset is included in this repository.

---

## State Representations

- **SR1 (lagged):** `x_k = [y_k, y_{k-1}, ..., y_{k-m}]`
- **SR2 (incremental):** `x_k = [y_k, Δy_k, ..., Δᵐy_k]`  

---

## Citation

If you use this code, please cite:

```bibtex
@misc{kececi2026xfodeexplainablefuzzyadditive,
      title={xFODE: An Explainable Fuzzy Additive ODE Framework for System Identification}, 
      author={Ertugrul Kececi and Tufan Kumbasar},
      year={2026},
      eprint={2604.14883},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.14883}, 
}
```
