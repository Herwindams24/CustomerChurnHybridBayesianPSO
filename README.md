# Customer Churn Prediction with a Hybrid Bayesian–PSO (HBP) Framework

![License](https://img.shields.io/badge/License-MIT-green)

A two-stage hyperparameter optimization framework that couples **Particle Swarm Optimization (PSO)** with **Bayesian Optimization (BO-TPE)** to tune deep neural networks (DNNs) for **customer churn prediction on class-imbalanced datasets**. The repository accompanies the manuscript currently under review and contains the full experimental notebook needed to reproduce the results.

---

## Overview

Churn prediction is a hard problem because the target class is rare (imbalanced) and DNN performance is highly sensitive to hyperparameter choices. **HBP (Hybrid Bayesian–PSO)** addresses both issues with a sequential, two-phase search:

- **Phase 1 — PSO (global exploration):** a swarm of **20 particles** runs for **30 iterations** to scan the hyperparameter space broadly and surface the **top-5 candidate regions**.
- **Phase 2 — Bayesian Optimization via Optuna TPE (local exploitation):** **50 trials** refine the promising regions identified by PSO to converge on a strong final configuration.

To handle imbalance without leaking information, **SMOTE–Tomek Links resampling is applied strictly inside each cross-validation fold (fold-internal resampling)**, so the validation data never sees synthetic samples generated from itself.

---

## Key Contributions

- **K1 — Sequential HBP efficacy:** demonstrates that PSO global search followed by Bayesian local refinement is more effective than either optimizer alone.
- **K2 — Fold-internal resampling integration:** integrates SMOTE–Tomek inside the CV loop to prevent data leakage during imbalanced learning.
- **K3 — Cross-domain generalizability:** validates the framework on two independent domains (banking and telecommunications).

---

## Experimental Design

Four conditions are compared under an identical evaluation protocol:

| Condition | Method | Role |
|-----------|--------|------|
| **C1** | Manual baseline | Reference configuration |
| **C2** | Standalone PSO | Global search only |
| **C3** | Standalone BO-TPE | Local search only |
| **C4** | **Hybrid HBP** | PSO → BO-TPE (proposed) |

### Datasets

| ID | Domain | Approx. size | Source |
|----|--------|--------------|---------|
| **K03** | Banking | ~10,000 samples | https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling |
| **K04** | Telecommunications | ~7,000 samples | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |

### Statistical Analysis

Results are compared with a **Friedman test → Wilcoxon signed-rank post-hoc → Holm–Bonferroni correction**, reporting **rank-biserial effect sizes**.

---

## Key Findings

Across **both datasets**, the proposed hybrid **C4 (HBP)** achieves the best **AUC**, **F1@opt**, **F1-min**, and **Friedman rank**, while also producing the **lowest standard deviation** (most stable runs) and the **most parsimonious architecture**.

> **Note on significance:** under the single-seed protocol used here, C4's margin over the standalone optimizers (C2/C3) is **not statistically significant**. The distinctive strength of HBP is therefore its **combination of competitive performance with greater stability and parsimony**. Extending to multi-seed experiments is identified as future work to support formal significance claims.

Exact per-metric values are reported in the manuscript inside the [`Paper/`](Paper/) folder.

---

## Repository Structure

```
.
├── Paper/            # Manuscript and related documents
├── notebook.ipynb    # End-to-end experiment: data prep, HBP, evaluation, stats
└── LICENSE           # MIT License
```

---

## Tech Stack

- **TensorFlow 2.20 / Keras 3** — DNN model
- **pyswarms** — Particle Swarm Optimization (Phase 1)
- **Optuna (TPE)** — Bayesian Optimization (Phase 2)
- **imbalanced-learn** — SMOTE–Tomek Links resampling
- **scikit-learn / SciPy** — cross-validation and statistical tests
- **Matplotlib** — publication-quality figures
- Reproducibility: fixed `seed = 42`, run on **Kaggle / Google Colab GPU (T4 / P100)**

---

## Getting Started

### Option A — Run in the cloud (recommended)

Open `notebook.ipynb` directly in **Google Colab** or **Kaggle** and select a GPU runtime. The notebook installs its own dependencies and runs end to end.

### Option B — Run locally

```bash
# clone
git clone https://github.com/Herwindams24/CustomerChurnHybridBayesianPSO.git
cd CustomerChurnHybridBayesianPSO

# (optional) create an environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
pip install tensorflow==2.20 optuna pyswarms imbalanced-learn scikit-learn scipy matplotlib pandas numpy jupyter

# launch
jupyter notebook notebook.ipynb
```

A GPU is strongly recommended; the full four-condition sweep across both datasets is compute-intensive on CPU.

---

## Reproducing the Results

1. Open `notebook.ipynb`.
2. Run all cells in order. The notebook:
   - loads and preprocesses the K03 (banking) and K04 (telco) datasets,
   - applies fold-internal SMOTE–Tomek resampling,
   - runs all four conditions (C1–C4),
   - performs the Friedman → Wilcoxon → Holm–Bonferroni analysis,
   - and exports the performance tables and figures.

The fixed seed ensures runs are reproducible.

---

## Citation

This work is currently **under peer review**. A formal citation (BibTeX) will be added here once the paper is published. In the meantime, please cite the repository:

```bibtex
@misc{customerchurn_hbp,
  title  = {Customer Churn Prediction with a Hybrid Bayesian--PSO (HBP) Framework},
  year   = {2025},
  note   = {Manuscript under review},
  howpublished = {\url{https://github.com/Herwindams24/CustomerChurnHybridBayesianPSO}}
}
```

---

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
