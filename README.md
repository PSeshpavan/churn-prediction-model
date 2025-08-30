
---

## 🚀 Quick Start

1. **Clone the repository** and install dependencies:
    ```sh
    git clone <repo-url>
    cd churn-prediction-model
    pip install -U pandas numpy scikit-learn shap xgboost joblib matplotlib
    ```

2. **Place your data**  
   Ensure `data.csv` is present in the root directory. The expected columns include customer demographics, account info, and a binary churn label (`Exited`).

3. **Run the pipeline**  
   Open and execute all cells in [`churn_prediction.ipynb`](churn_prediction.ipynb)for a full, reproducible workflow:
   - Data loading & preprocessing
   - Model training & selection (Logistic Regression, Random Forest, XGBoost if available)
   - Evaluation (metrics, ROC, confusion matrix)
   - Explainability (feature importance, SHAP)
   - Export of all artifacts for Tableau

---

## 📊 Tableau Integration

- **Outputs for Tableau** are saved in the `artifacts/` folder:
    - `tableau_export.csv` — compact predictions for Tableau dashboards
    - `predictions.csv` — full predictions with helper columns
    - `metrics.csv` — test set metrics (accuracy, precision, recall, f1, ROC AUC)
    - `feature_importance.csv` — ranked feature importances
    - `shap_summary.png`, `shap_bar_top20.png` — SHAP explainability plots

- **How to use in Tableau:**
    1. Open Tableau Desktop/Public.
    2. Connect to `artifacts/tableau_export.csv` (or `predictions.csv` for more detail).
    3. Suggested visuals:
        - Churn probability distribution (histogram of `Probability`)
        - Churn by top feature (bar: `TopFeature` vs avg `Probability`)
        - Confusion KPIs (precision, recall, f1 from `metrics.csv`)

---

## 🧠 Notebooks Overview

- [`churn.ipynb`](churn.ipynb):  
  Exploratory notebook with classic multi-model training, manual feature importance, and evaluation.

- [`churn_prediction.ipynb`](churn_prediction.ipynb):  
  Production-style pipeline with robust preprocessing, best-model selection, explainability, and automated Tableau artifact export.

---

## 📁 Artifacts

All model outputs and Tableau-ready files are saved to the [`artifacts/`](artifacts/) directory:

- `model.pkl` — Trained pipeline (preprocessing + best model)
- `metrics.csv` — Key test metrics
- `feature_importance.csv` — Model feature importance
- `feature_importance_top20.png` — Top 20 features chart
- `roc_curve.png` — ROC curve (binary target)
- `shap_summary.png` — Global SHAP summary plot
- `shap_bar_top20.png` — Top 20 SHAP features bar plot
- `predictions.csv` — Full predictions with helper columns
- `tableau_export.csv` — Compact predictions for Tableau

---

## 📝 Customization

- **Change the target column:**  
  If your churn label is not detected automatically, set `target_col` manually in the notebook after loading the data.

- **Add new models:**  
  Edit the `models` list in [`churn_prediction.ipynb`](churn_prediction.ipynb) to include additional classifiers.

- **Feature engineering:**  
  Add new features or transformations in the preprocessing section of the notebook.

---

## 🛠️ Requirements

- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, shap, joblib
- (Optional) xgboost for XGBoost model support

---

## 📈 Tableau Dashboard

**Dashboard Link:**  
*Add your Tableau Public/Server dashboard link here once published.*

---

## 📄 License

*Specify your license here (e.g., MIT, Apache 2.0, etc.)*

---

## 🤝 Acknowledgements

- Dataset inspired by [Kaggle Churn Datasets](https://www.kaggle.com/)
- SHAP explainability: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- Pipeline design inspired by best practices in ML Ops and analytics
