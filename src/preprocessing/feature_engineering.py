import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
FINAL_PATH   = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"
FEATS_PATH   = PROJECT_ROOT / "models" / "trained_models" / "selected_features.txt"



df = pd.read_csv(CLEANED_PATH)
X_raw = df.drop("Churn", axis=1)
y = df["Churn"].astype(int)

print(f"Original features: {X_raw.shape[1]}")


charge_cols = [c for c in X_raw.columns if "charge" in c.lower()]
X = X_raw.drop(columns=charge_cols)
print(f"Dropped {len(charge_cols)} charge columns (redundant with minutes)")


print("Creating advanced features...")

# 1. Customer tenure (from Account length in days → months)
X["Customer_tenure_months"] = X["Account length"] / 30.0

# 2. Total minutes & calls
X["Total_Minutes"] = X[["Total day minutes", "Total eve minutes", "Total night minutes", "Total intl minutes"]].sum(axis=1)
X["Total_Calls"] = X[["Total day calls", "Total eve calls", "Total night calls", "Total intl calls"]].sum(axis=1)

# 3. Usage patterns
X["Avg_Daily_Usage"] = X["Total_Minutes"] / (X["Customer_tenure_months"] + 1)
X["Call_Frequency"] = X["Total_Calls"] / (X["Customer_tenure_months"] + 1)
X["Intl_Usage_Rate"] = X["Total intl minutes"] / (X["Total_Minutes"] + 1)

# 4. High engagement flags
X["High_Service_Calls"] = (X["Customer service calls"] > 3).astype(int)
X["Has_Vmail"] = (X["Number vmail messages"] > 0).astype(int)

# 5. Log transform skewed features
log_features = ["Total_Minutes", "Total_Calls", "Avg_Daily_Usage", "Call_Frequency"]
for col in log_features:
    if col in X.columns:
        X[f"log_{col}"] = np.log1p(X[col])

print(f"Engineered {len(log_features)} log features + 7 new ones")


sig_num, sig_cat = [], []

# T-test
num_cols = X.select_dtypes(include=[np.number]).columns
for col in num_cols:
    t, p = stats.ttest_ind(X[col][y==1], X[col][y==0], equal_var=False, nan_policy='omit')
    if p < 0.05:
        sig_num.append(col)

# Chi-squared
cat_cols = [c for c in X.columns if any(prefix in c for prefix in ['State_', 'plan_Yes', 'Area code'])]
for col in cat_cols:
    tab = pd.crosstab(X[col], y)
    chi2, p, _, _ = chi2_contingency(tab)
    if p < 0.05:
        sig_cat.append(col)


print("Running RFE on cleaned features...")
rfe = RFE(RandomForestClassifier(n_estimators=200, random_state=42), n_features_to_select=20)
rfe.fit(X, y)
selected = X.columns[rfe.support_].tolist()


df_final = X[selected].copy()
df_final["Churn"] = y
df_final.to_csv(FINAL_PATH, index=False)

with open(FEATS_PATH, "w") as f:
    f.write("\n".join(selected))

print(f"\nMILESTONE 2 COMPLETE")
print(f"   → Final data: {FINAL_PATH}")
print(f"   → Selected: {len(selected)} features")