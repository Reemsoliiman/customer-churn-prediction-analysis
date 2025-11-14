import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_processed_data.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Ensure no missing values
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test data
joblib.dump((X_test, y_test), MODEL_DIR / "test_data.pkl")
print("Test data saved")

# DO NOT SAVE PREPROCESSOR HERE — already handled in pipeline
# We train directly on selected features

models = {
    "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss'
    )
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  → Saved: {path.name}")

print("\nALL MODELS TRAINED & SAVED")