import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

# Load dataset
df = pd.read_csv("data/interim/cleaned.csv", low_memory=False)

# Remove negative claims and create 'has_claim' flag
df = df[df["TotalClaims"] >= 0]
df["has_claim"] = (df["TotalClaims"] > 0).astype(int)
df["PostalCode"] = df["PostalCode"].astype(str)

# Drop unnecessary columns
drop_cols = ["PolicyID", "UnderwrittenCoverID", "TransactionMonth"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Separate categorical and numerical features
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
for col in ["TotalClaims", "has_claim"]:
    if col in num_cols:
        num_cols.remove(col)

# Preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# ================= Claim Severity Regression =================
severity_df = df[df["TotalClaims"] > 0]

if len(severity_df) == 0:
    print("⚠ No positive claims found — skipping severity model.")
    reg_results = None
    best_reg_model = None
else:
    X_sev = severity_df.drop("TotalClaims", axis=1)
    y_sev = severity_df["TotalClaims"]

    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=42
    )

    models_reg = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBRegressor": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="rmse"
        )
    }

    reg_results = {}
    print("\n================ Claim Severity Regression ================\n")

    for name, model in models_reg.items():
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X_train_sev, y_train_sev)

        preds = pipe.predict(X_test_sev)
        rmse = mean_squared_error(y_test_sev, preds, squared=False)
        r2 = r2_score(y_test_sev, preds)
        reg_results[name] = {"RMSE": rmse, "R2": r2}
        print(f"{name}: RMSE={rmse:.2f} | R2={r2:.4f}")

    best_reg_name = min(reg_results, key=lambda n: reg_results[n]["RMSE"])
    best_reg_model = Pipeline([("prep", preprocess), ("model", models_reg[best_reg_name])])
    best_reg_model.fit(X_train_sev, y_train_sev)

# ================= Claim Probability Classification =================
X_cls = df.drop("has_claim", axis=1)
y_cls = df["has_claim"]

# Skip classification if all claims are 0
if y_cls.nunique() < 2:
    print("⚠ Not enough positive/negative samples — cannot train classification model.")
    cls_results = None
    best_cls_model = None
else:
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )

    models_cls = {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBClassifier": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.9
        )
    }

    cls_results = {}
    print("\n================ Claim Probability Classification ================\n")

    for name, model in models_cls.items():
        pipe = Pipeline([("prep", preprocess), ("model", model)])
        pipe.fit(X_train_cls, y_train_cls)

        preds = pipe.predict(X_test_cls)

        accuracy = accuracy_score(y_test_cls, preds)
        precision = precision_score(y_test_cls, preds, zero_division=0)
        recall = recall_score(y_test_cls, preds, zero_division=0)
        f1 = f1_score(y_test_cls, preds, zero_division=0)

        cls_results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

        print(f"{name}: Acc={accuracy:.4f} | Prec={precision:.4f} | Rec={recall:.4f} | F1={f1:.4f}")

    best_cls_name = max(cls_results, key=lambda n: cls_results[n]["F1"])
    best_cls_model = Pipeline([("prep", preprocess), ("model", models_cls[best_cls_name])])
    best_cls_model.fit(X_train_cls, y_train_cls)

# ================= SHAP Feature Importance =================
if best_reg_model:
    print("\n================ SHAP Feature Importance (Severity Model) ================\n")
    X_train_transformed = best_reg_model.named_steps["prep"].transform(X_sev)
    explainer = shap.Explainer(best_reg_model.named_steps["model"], X_train_transformed)
    shap_values = explainer(X_train_transformed)
    print("SHAP analysis computed for severity model.")

if best_cls_model:
    print("\n================ SHAP Feature Importance (Classification Model) ================\n")
    X_train_cls_transformed = best_cls_model.named_steps["prep"].transform(X_train_cls)
    explainer_cls = shap.Explainer(best_cls_model.named_steps["model"], X_train_cls_transformed)
    shap_values_cls = explainer_cls(X_train_cls_transformed)
    print("SHAP analysis computed for classification model.")

# ================= MODEL PERFORMANCE SUMMARY =================
if reg_results:
    print("\n--- Severity Model Results ---")
    for name, r in reg_results.items():
        print(f"{name}: RMSE={r['RMSE']:.2f} | R2={r['R2']:.4f}")
    print(f"\nBest Severity Model: {best_reg_name}\n")

if cls_results:
    print("\n--- Claim Probability Model Results ---")
    for name, r in cls_results.items():
        print(f"{name}: Acc={r['Accuracy']:.4f} | F1={r['F1']:.4f}")
    print(f"\nBest Classifier: {best_cls_name}\n")

print("\nTask-4 modeling complete.")
