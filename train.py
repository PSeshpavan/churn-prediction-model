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

DATA_PATH = "data.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load data
df = pd.read_csv(DATA_PATH)
target_col = "Exited"

# Separate features/target
X = df.drop(columns=[target_col])
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

# Train RF
clf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
pipe.fit(X_train, y_train)

# Save the fitted pipeline
model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
joblib.dump(pipe, model_path)
print(f"Model retrained and saved to {model_path}")
