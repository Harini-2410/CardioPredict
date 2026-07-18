"""
Step 4: Generate training curves + confusion matrix plot for the report
Saves plots into ../outputs/
"""
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

history = joblib.load("training_history.pkl")
model = tf.keras.models.load_model("cardiopredict_ann.keras")
data = np.load("splits.npz")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history["accuracy"], label="Train Accuracy")
axes[0].plot(history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Model Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history["loss"], label="Train Loss")
axes[1].plot(history["val_loss"], label="Val Loss")
axes[1].set_title("Model Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../outputs/training_curves.png", dpi=150)
plt.close()

y_pred_proba = model.predict(data["X_test"], verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)
cm = confusion_matrix(data["y_test"], y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix - Test Set")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("../outputs/confusion_matrix.png", dpi=150)
plt.close()

print("✅ Saved ../outputs/training_curves.png and ../outputs/confusion_matrix.png")