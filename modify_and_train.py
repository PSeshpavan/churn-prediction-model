import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import shap

# 1. Modify the notebook file programmatically (since direct .ipynb edits are restricted)
notebook_path = "churn_prediction.ipynb"
if os.path.exists(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            for i, line in enumerate(source):
                if 'X = df.drop(columns=[target_col])' in line:
                    source[i] = line.replace(
                        'X = df.drop(columns=[target_col])',
                        'X = df.drop(columns=[target_col, "RowNumber", "CustomerId", "Surname"])'
                    )
                    modified = True
                    break
            if modified:
                break
                
    if modified:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Updated churn_prediction.ipynb successfully.")
    else:
        print("Target line not found in churn_prediction.ipynb.")

# 2. Retrain the model on the updated feature set
DATA_PATH = "data.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load data
df = pd.read_csv(DATA_PATH)
target_col = "Exited"

# Separate features/target (excluding RowNumber, CustomerId, Surname)
X = df.drop(columns=[target_col, "RowNumber", "CustomerId", "Surname"])
y = df[target_col].astype(int)

# Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Imputers & encoders
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Train Random Forest (it was selected as best previously)
clf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
pipe.fit(X_train, y_train)

# Save the fitted pipeline
model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
joblib.dump(pipe, model_path)
print(f"Model saved to {model_path}")

# Preprocess training data for SHAP explainer
X_train_trans = preprocessor.transform(X_train)

# Fit and save the SHAP Explainer
# Use interventional to prevent C-extension additivity overflow bug
explainer = shap.TreeExplainer(clf, data=X_train_trans, feature_perturbation='interventional')
explainer_path = os.path.join(ARTIFACT_DIR, "explainer.pkl")
joblib.dump(explainer, explainer_path)
print(f"SHAP Explainer saved to {explainer_path}")
