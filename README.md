# TOC_PREDICTION

A reproducible benchmark for **TOC (Total Organic Carbon) prediction** using multiple machine learning models on a unified dataset.

This repository provides a simple and consistent pipeline for model training, evaluation, visualization, and result export. It is designed for **experimental comparison**, **baseline reproduction**, and **regression benchmarking** on tabular TOC prediction tasks.

---

## Highlights

- Unified workflow for multiple regression models
- PSO-based hyperparameter optimization
- Standard regression metrics for fair comparison
- Automatic saving of predictions, figures, and trained models
- Easy to reproduce and extend

---

## Implemented Models

The current benchmark includes:

- **PSO-RF** — Random Forest optimized by Particle Swarm Optimization
- **PSO-SVM** — Support Vector Machine optimized by Particle Swarm Optimization
- **PSO-LightGBM** — LightGBM optimized by Particle Swarm Optimization
- **TabPFN** — TabPFN regressor with default settings

---

## Repository Structure

```bash
.
├── TOCPRE.py               # Main benchmark script
├── WXN2.csv                # Input dataset
├── requirements.txt        # Python dependencies
└── all_model_results/      # Output directory generated after running
```

---

## Dataset

The script reads the dataset from:

```bash
./WXN2.csv
```

By default:

- columns `0:14` are used as input features
- column `14` is used as the target value

Please place the dataset in the project root before running the script.

---

## Environment

Recommended:

- Python 3.9+

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Quick Start

Run the benchmark with:

```bash
python TOCPRE.py
```

After execution, all generated results will be saved to:

```bash
./all_model_results
```

---

## Output Files

The script automatically generates and saves:

- **Prediction files**  
  e.g. `PSO_RF_predictions.csv`, `PSO_SVM_predictions.csv`

- **Model files**  
  e.g. `pso_rf_model.pkl`, `pso_svm_model.pkl`, `tabpfn_model.pkl`

- **Visualization files**  
  e.g. true-vs-predicted plots, optimization curves, model comparison plots

- **Summary tables**  
  - `model_summary.csv`
  - `model_details.csv`

---

## Evaluation Metrics

The benchmark reports the following regression metrics:

- **R²**
- **MSE**
- **RMSE**
- **MAE**

Both training and test performance are recorded for comparison.

---

## Reproducibility

This repository uses:

- a fixed random seed
- a unified train/test split
- a consistent evaluation pipeline across all models

These settings are intended to improve reproducibility and ensure fair comparison.

---

## Notes

- `lightgbm` is an optional dependency. If it is not installed, the PSO-LightGBM part will be skipped.
- `tabpfn` is an optional dependency. If it is not installed, the TabPFN part will be skipped.
- In the current implementation, SVM uses feature scaling and target scaling before training.
- All outputs are saved automatically, which makes the script convenient for result analysis and paper figures.

---

## Use Cases

This repository can be used for:

- TOC prediction experiments
- baseline comparison studies
- regression model benchmarking
- parameter optimization experiments
- reproducible result generation for academic work

---

## Future Improvements

Possible extensions include:

- adding more regression baselines
- supporting cross-validation
- adding feature importance analysis
- integrating deep learning models for tabular regression
- supporting custom datasets through command-line arguments

---

## Citation

If you use this repository in academic research, please cite the related paper or describe this repository as the implementation source of your benchmark experiments.

---

## License

This project is intended for research and academic use.

