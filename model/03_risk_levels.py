"""
Step 3: Convert binary probability output -> Low / Medium / High risk levels
+ simple recommendation engine
"""
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

model = tf.keras.models.load_model("cardiopredict_ann.keras")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")

data = np.load("splits.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Get probabilities
probs = model.predict(X_test, verbose=0).flatten()

# Bucket into 3 risk tiers using QUANTILES of the training probability
# distribution, not fixed 0.33/0.66 cutoffs.
# Why: this model separates classes almost perfectly (near-0 or near-1
# probabilities), so a fixed 0.33/0.66 split leaves Medium nearly empty.
train_probs = model.predict(data["X_train"], verbose=0).flatten()
q33, q66 = np.quantile(train_probs, [0.3333, 0.6667])
print(f"Risk thresholds (from training distribution): q33={q33:.4f}, q66={q66:.4f}")

def risk_bucket(p, low_thresh=q33, high_thresh=q66):
    if p < low_thresh:
        return "Low"
    elif p < high_thresh:
        return "Medium"
    else:
        return "High"

risk_labels = np.array([risk_bucket(p) for p in probs])
joblib.dump({"q33": q33, "q66": q66}, "risk_thresholds.pkl")

# Distribution check
unique, counts = np.unique(risk_labels, return_counts=True)
print("Risk level distribution on test set:")
for u, c in zip(unique, counts):
    print(f"  {u}: {c} ({c/len(risk_labels)*100:.1f}%)")

# Recommendation engine
def get_recommendation(risk_level):
    recommendations = {
        "Low": "Low risk detected. Maintain a healthy diet, stay active, and go for annual checkups.",
        "Medium": "Medium risk detected. Consider consulting a doctor and monitor blood pressure, "
                  "cholesterol, and blood sugar regularly.",
        "High": "High risk detected. Please consult a cardiologist as soon as possible for a full evaluation."
    }
    return recommendations[risk_level]

print("\nSample predictions:")
for i in range(5):
    print(f"  Probability: {probs[i]:.3f} -> Risk: {risk_labels[i]} "
          f"| Actual label: {'Disease' if y_test[i]==1 else 'No Disease'}")
    print(f"    Recommendation: {get_recommendation(risk_labels[i])}")

def predict_risk(patient_dict):
    df_input = pd.DataFrame([patient_dict])
    df_encoded = pd.get_dummies(df_input)

    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_cols]

    X_scaled = scaler.transform(df_encoded)
    prob = float(model.predict(X_scaled, verbose=0).flatten()[0])
    thresholds = joblib.load("risk_thresholds.pkl")
    risk = risk_bucket(prob, thresholds["q33"], thresholds["q66"])

    return {
        "probability": round(prob, 4),
        "risk_level": risk,
        "recommendation": get_recommendation(risk)
    }

if __name__ == "__main__":
    example_patient = {
        "Age": 62, "Gender": "Male", "Weight": 90, "Height": 172, "BMI": 30.4,
        "Smoking": "Current", "Alcohol_Intake": "High", "Physical_Activity": "Sedentary",
        "Diet": "Unhealthy", "Stress_Level": "High", "Hypertension": 1, "Diabetes": 1,
        "Hyperlipidemia": 1, "Family_History": 1, "Previous_Heart_Attack": 0,
        "Systolic_BP": 150, "Diastolic_BP": 95, "Heart_Rate": 88,
        "Blood_Sugar_Fasting": 140, "Cholesterol_Total": 260
    }
    result = predict_risk(example_patient)
    print("\n" + "=" * 60)
    print("EXAMPLE PATIENT PREDICTION")
    print("=" * 60)
    print(result)