"""
Step 1: Preprocessing for CardioPredict ANN
Run this from inside the model/ folder: python3 01_preprocess.py
Reads dataset from ../data/, saves outputs into model/ (current folder)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("../data/heart_disease.csv")
print("Shape:", df.shape)

# 1. Handle missing values
df["Alcohol_Intake"] = df["Alcohol_Intake"].fillna("None")

# 2. Split feature types
target_col = "Heart_Disease"
categorical_cols = ["Gender", "Smoking", "Alcohol_Intake", "Physical_Activity", "Diet", "Stress_Level"]

# 3. One-hot encode categoricals
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

feature_cols = [c for c in df_encoded.columns if c != target_col]
X = df_encoded[feature_cols]
y = df_encoded[target_col]

print("Final feature count:", X.shape[1])

# 4. Train / val / test split (70/15/15), stratified
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. Save everything (into current folder = model/)
np.savez("splits.npz",
         X_train=X_train_scaled, y_train=y_train.values,
         X_val=X_val_scaled, y_val=y_val.values,
         X_test=X_test_scaled, y_test=y_test.values)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

print("\n✅ Preprocessing complete. Saved: splits.npz, scaler.pkl, feature_cols.pkl")