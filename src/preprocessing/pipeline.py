import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "merged_churn_data.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
PREPROC_PATH = PROJECT_ROOT / "models" / "trained_models" / "preprocessor.pkl"

# Create directories
(PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "models" / "trained_models").mkdir(parents=True, exist_ok=True)


print(f"Loading raw data from: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)
print(f"Original shape: {df.shape}")


n_before = len(df)
df = df.drop_duplicates()
print(f"Removed {n_before - len(df)} duplicate rows")


print("Imputing missing values...")

# Numerical → median
num_cols = df.select_dtypes(include=[np.number]).columns
if df[num_cols].isnull().any().any():
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical → mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled")


print("Creating features...")

# Basic totals
df["Total_Minutes"] = df[["Total day minutes", "Total eve minutes", "Total night minutes", "Total intl minutes"]].sum(axis=1)
df["Total_Charge"]  = df[["Total day charge", "Total eve charge", "Total night charge", "Total intl charge"]].sum(axis=1)
df["High_Customer_Service"] = (df["Customer service calls"] > 3).astype(int)

# Interaction features
df["Intl_Minutes_per_Call"] = df["Total intl minutes"] / (df["Total intl calls"] + 1)
df["High_Charge"] = (df["Total_Charge"] > df["Total_Charge"].median()).astype(int)
df["High_Charge_High_Service"] = df["High_Charge"] * df["High_Customer_Service"]
df["Vmail_Rate"] = df["Number vmail messages"] / (df["Total_Minutes"] + 1)
df["Day_Night_Ratio"] = df["Total day minutes"] / (df["Total night minutes"] + 1)

print(f"Engineered {len(df.columns) - 20} new features")

print("Clipping outliers...")

def clip_iqr(series, factor=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.clip(q1 - factor * iqr, q3 + factor * iqr)

numerical_cols = df.select_dtypes(include=[np.number]).columns.drop("Churn", errors="ignore")
for col in numerical_cols:
    df[col] = clip_iqr(df[col])


X = df.drop("Churn", axis=1)
y = df["Churn"].astype(int)

categorical_features = ["State", "International plan", "Voice mail plan"]
numerical_features = [c for c in X.columns if c not in categorical_features]

print(f"Encoding: {categorical_features}")
print(f"Scaling: {len(numerical_features)} numerical features")


preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
], remainder="drop")

# Fit and transform
X_transformed = preprocessor.fit_transform(X)

# Get final feature names
ohe = preprocessor.named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)
final_feature_names = numerical_features + list(cat_feature_names)


df_final = pd.DataFrame(X_transformed, columns=final_feature_names)
df_final["Churn"] = y.values


df_final.to_csv(CLEANED_PATH, index=False)
joblib.dump(preprocessor, PREPROC_PATH)

print(f"\nMILESTONE 1 COMPLETE")
print(f"   → CLEANED DATA: {CLEANED_PATH}")
print(f"   → Shape: {df_final.shape}")
print(f"   → Features: {len(final_feature_names)}")
print(f"   → Preprocessor saved: {PREPROC_PATH}")