# ❤️ CardioPredict

**A deep learning powered heart disease risk screening application.**

CardioPredict uses an Artificial Neural Network (ANN) trained on patient health data to estimate heart disease risk — classifying users into **Low**, **Medium**, or **High** risk tiers, with tailored health recommendations and a nearby-hospital finder for higher-risk results.

> ⚠️ This is an academic project and preliminary screening tool only. It is **not** a medical diagnosis and does not replace professional healthcare advice.

---

## 🚀 Features

- 🧠 **Deep learning risk prediction** — ANN trained on 50,000 patient records, 99.8%+ test accuracy
- 🎯 **3-tier risk classification** — Low / Medium / High, using quantile-based thresholds tuned to the model's actual probability distribution
- 🔐 **User authentication** — signup/login with hashed passwords (SQLite + Werkzeug security)
- 🏥 **Nearby hospital finder** — geolocation + OpenStreetMap integration, shown automatically for Medium/High risk results
- 💡 **Personalized recommendations** — actionable health guidance based on risk level
- 🎨 **Polished, animated UI** — light-themed interface with smooth transitions and a live risk-spectrum visualization

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌───────────────────┐
│    Frontend      │  HTTP   │   Flask Backend   │         │   Trained ANN       │
│  (HTML/CSS/JS)    │──────▶ │      (app.py)      │──────▶ │  (Keras/TensorFlow)  │
│  login/signup/     │◀────── │  /predict          │◀────── │  cardiopredict_ann   │
│  risk form          │         │  /signup /login     │         │  .keras               │
└─────────────────┘         └──────────────────┘         └───────────────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   SQLite (users.db) │
                              └──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras (ANN — Dense layers with BatchNorm + Dropout) |
| Backend | Flask, Flask-CORS, Werkzeug (password hashing) |
| Database | SQLite |
| Frontend | Vanilla HTML/CSS/JavaScript (no build step) |
| Data processing | pandas, scikit-learn |
| Visualization | matplotlib, seaborn |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 99.85% |
| F1 Score | 99.84% |
| ROC-AUC | 1.0000 |

**Confusion Matrix (test set, n=7,500):**

| | Predicted: No Disease | Predicted: Disease |
|---|---|---|
| **Actual: No Disease** | 4019 | 5 |
| **Actual: Disease** | 6 | 3470 |

> Note: This dataset is synthetic. Near-perfect accuracy reflects clean, rule-generated data rather than real-world clinical noise — see [Limitations](#-limitations--academic-notes) below.

---

## 📁 Project Structure

```
CardioPredict/
├── data/
│   └── heart_disease.csv          # dataset (not tracked in git)
├── model/
│   ├── 01_preprocess.py           # cleaning, encoding, train/val/test split
│   ├── 02_train_model.py          # ANN training
│   ├── 03_risk_levels.py          # quantile-based risk bucketing
│   ├── 04_plots.py                # training curve + confusion matrix generation
│   ├── cardiopredict_ann.keras    # trained model
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   └── risk_thresholds.pkl
├── backend/
│   └── app.py                     # Flask API (predict, signup, login)
├── frontend/
│   ├── login.html
│   ├── signup.html
│   └── index.html                 # main risk assessment form
├── outputs/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Harini-2410/CardioPredict.git
cd CardioPredict
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place your `heart_disease.csv` in the `data/` folder. Dataset used: [Synthetic Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/arifmia/heat-disease-predication-dataset) (Kaggle).

### 4. Train the model (one-time)
```bash
cd model
python 01_preprocess.py
python 02_train_model.py
python 03_risk_levels.py
python 04_plots.py
```

### 5. Start the backend
```bash
cd ../backend
python app.py
```
Runs at `http://127.0.0.1:5000`

### 6. Serve the frontend
```bash
cd ../frontend
python -m http.server 5500
```
Open `http://localhost:5500/login.html` in your browser.

---

## 🧪 Usage

1. Sign up for an account (or log in if you already have one)
2. Fill in the health assessment form — demographics, vitals, lifestyle, medical history
3. Click **Assess Risk** to get an instant prediction
4. For Medium/High risk results, use **Find Nearby Hospitals** to locate care near you

---

## ⚠️ Limitations & Academic Notes

- **Architecture choice:** Uses a feedforward ANN (Dense layers), not LSTM/CNN — appropriate since the data is flat tabular data with no sequential or spatial structure.
- **Synthetic dataset:** The near-perfect accuracy reflects a synthetic dataset generated from clean underlying rules. Real clinical data would show more class overlap and lower achievable accuracy.
- **Risk-level bucketing:** Uses quantile-based thresholds (not fixed 0.33/0.66 cutoffs) since the model's probability outputs are strongly bimodal.
- **Authentication:** Uses browser `localStorage` for session state — suitable for an academic demo, not production-grade session security.
- **Hospital finder:** Uses OpenStreetMap community data, which may have gaps in some regions compared to commercial mapping services.

---

## 👩‍💻 Author

**Harini** — B.Tech IT, Malla Reddy University

---

## 📄 License

This project was built for academic purposes as part of a deep learning coursework project.