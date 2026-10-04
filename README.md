# Landslide Prediction System

A machine learning research project exploring landslide classification from environmental and terrain features, with early warning on edge devices as a future goal. This repository contains saved model artifacts, evaluation reports, plots, and data preparation scripts for Sikkim and Kerala, India.

## Project status

The repository currently provides experiment outputs and data preparation utilities. The training script referenced in the project notes (`train_landslide_models.py`), the original Kaggle training CSV, a live prediction application, sensor integration, and edge deployment code are not included. The saved training run cannot be reproduced end to end from this checkout alone.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `Explanation.txt` | Detailed project explanation and description of the original experiment |
| `summary.json` | Recorded dataset information, model scores, cross-validation results, and timings |
| `model_comparison.csv`, `cross_validation_results.csv` | Tabular evaluation results |
| `dataset_summary.txt`, `classification_report_*.txt` | Dataset statistics and classification reports |
| `logistic_regression.pkl`, `random_forest.pkl`, `xgboost.pkl`, `lightgbm.pkl` | Saved model artifacts |
| `scaler.pkl` | Saved preprocessing scaler |
| Root-level PNG files | Confusion matrices, ROC and precision-recall curves, learning curves, feature importance, and SHAP plots |
| `Real_data(NOT SYNTHETIC)/` | Inventory CSVs, prepared datasets, and three data preparation scripts |
| `_review_deck_render/` | Supporting presentation image |

## Recorded experiment

The saved reports describe a run on the Kaggle Landslide Dataset by `rajumavinmar`. `Explanation.txt` identifies this dataset as synthetic. The run used 2,000 rows, nine input features, and a balanced binary target (`Landslide`: 1 for landslide, 0 for no landslide).

Input features are rainfall, slope angle, soil saturation, vegetation cover, earthquake activity, proximity to water, and three soil-type indicators: gravel, sand, and silt. The notes describe an 80/20 stratified train/test split and a scaler fitted on training data.

| Model | Recorded hold-out accuracy | Recorded hold-out F1 | Recorded hold-out ROC AUC | Mean CV accuracy |
| --- | --- | --- | --- | --- |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 0.9995 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

These are previously recorded scores from `model_comparison.csv` and `summary.json`, not new validation results. They do not establish predictive performance on field data. `summary.json` names Logistic Regression as the best model for this run.

## Data preparation scripts

The directory name `Real_data(NOT SYNTHETIC)` does not mean every row and feature is directly observed. The scripts mix inventory data or listed event locations with generated samples, derived features, and fallback estimates.

| Script | Output | What it does |
| --- | --- | --- |
| `prepare_real_data.py` | `landslide_real_sikkim.csv` | Combines point and polygon inventories, generates an equal number of synthetic non-landslide samples, and encodes categorical features |
| `extract_real_data.py` | `landslide_real_enriched_sikkim.csv` | Enriches Sikkim inventory records with regional NASA POWER and USGS queries, estimates additional features, and generates non-landslide samples |
| `collect_kerala_data.py` | `kerala_landslide_real.csv` | Uses event coordinates listed in the script, expands nearby positive samples, selects non-event grid points, and queries climate, earthquake, and elevation services |

The scripts refer to the Sikkim inventory CSVs as Zenodo data; the Kerala script describes its event list as drawn from documented records. Exact source citations and redistribution terms should be verified against the original sources.

Missing Sikkim event years can be randomly assigned. API failures can trigger estimated or random fallback values. Some Kerala climate features are generated using the target label, which can introduce target leakage. Non-event grid locations are assumed negatives, not verified stable slopes. Use these datasets for exploratory work with those limitations in mind; their schemas also differ from the original model inputs.

## Getting started

Clone the repository and create a Python environment. The following commands use Windows PowerShell:

```powershell
git clone https://github.com/zoror2/LandSlide_prediction-System.git
cd LandSlide_prediction-System
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas requests
```

Before running a script, update its `FOLDER` constant to the absolute path of your local `Real_data(NOT SYNTHETIC)` directory. All three scripts currently hard-code the author's Windows path. Alternatively, replace that constant in each script with:

```python
FOLDER = os.path.dirname(os.path.abspath(__file__))
```

Run the desired preparation workflow from the repository root:

```powershell
# Prepare Sikkim terrain data from the included inventory CSVs.
python "Real_data(NOT SYNTHETIC)/prepare_real_data.py"

# Prepare Sikkim data with regional climate and earthquake features.
python "Real_data(NOT SYNTHETIC)/extract_real_data.py"

# Collect and prepare the Kerala dataset.
python "Real_data(NOT SYNTHETIC)/collect_kerala_data.py"
```

Each command regenerates its corresponding output CSV. The enrichment and Kerala collection scripts require internet access for their API queries and may take time to complete. Check their console output for fallback messages before interpreting the resulting data.

## Saved models and reproducibility

Model inspection or inference may require `scikit-learn`, `joblib`, `xgboost`, and `lightgbm`, in addition to the preparation dependencies. The repository does not record the original package versions or provide a complete inference workflow. Before using a saved model, establish its serialization format, expected feature order, preprocessing requirements, and compatible library versions. Load pickle artifacts only from a trusted source.

To reproduce the original experiment, the missing training code and original dataset must be supplied. To train on the regional datasets, adapt the feature schema and use validation that accounts for related locations and event years.

## Intended direction

The project notes propose low-power edge inference for landslide early warning. Field validation, reliable sensor inputs, deployment code, and alert integration remain future work.
