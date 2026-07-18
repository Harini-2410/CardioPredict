"""
CardioPredict API
Run this from inside the backend/ folder: python3 app.py
Loads trained model + artifacts from ../model/
Then POST to http://localhost:5000/predict
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import tensorflow as tf
import joblib
import sqlite3
import os

app = Flask(__name__)
CORS(app)  # allows your frontend (different origin) to call this API

MODEL_DIR = "../model/"
DB_PATH = "users.db"

model = tf.keras.models.load_model(MODEL_DIR + "cardiopredict_ann.keras")
scaler = joblib.load(MODEL_DIR + "scaler.pkl")
feature_cols = joblib.load(MODEL_DIR + "feature_cols.pkl")
thresholds = joblib.load(MODEL_DIR + "risk_thresholds.pkl")

# -----------------------------
# Database setup
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

REQUIRED_FIELDS = [
    "Age", "Gender", "Weight", "Height", "BMI", "Smoking", "Alcohol_Intake",
    "Physical_Activity", "Diet", "Stress_Level", "Hypertension", "Diabetes",
    "Hyperlipidemia", "Family_History", "Previous_Heart_Attack",
    "Systolic_BP", "Diastolic_BP", "Heart_Rate", "Blood_Sugar_Fasting",
    "Cholesterol_Total"
]

def risk_bucket(p):
    if p < thresholds["q33"]:
        return "Low"
    elif p < thresholds["q66"]:
        return "Medium"
    else:
        return "High"

def get_recommendation(risk_level):
    recommendations = {
        "Low": "Low risk detected. Maintain a healthy diet, stay active, and go for annual checkups.",
        "Medium": "Medium risk detected. Consider consulting a doctor and monitor blood pressure, "
                  "cholesterol, and blood sugar regularly.",
        "High": "High risk detected. Please consult a cardiologist as soon as possible for a full evaluation."
    }
    return recommendations[risk_level]

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are all required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return jsonify({"error": "An account with this email already exists."}), 409

        password_hash = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash)
        )
        conn.commit()
        return jsonify({"name": name, "email": email})
    finally:
        conn.close()

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"error": "Invalid email or password."}), 401

        return jsonify({"name": user["name"], "email": user["email"]})
    finally:
        conn.close()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        df_input = pd.DataFrame([data])[REQUIRED_FIELDS]
        df_encoded = pd.get_dummies(df_input)

        for col in feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_cols]

        X_scaled = scaler.transform(df_encoded)
        prob = float(model.predict(X_scaled, verbose=0).flatten()[0])
        risk = risk_bucket(prob)

        return jsonify({
            "probability": round(prob, 4),
            "risk_level": risk,
            "recommendation": get_recommendation(risk)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)