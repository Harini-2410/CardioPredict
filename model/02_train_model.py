"""
Step 2: Train ANN (deep learning) model on preprocessed data
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, f1_score)
import joblib

tf.random.set_seed(42)
np.random.seed(42)

# Load preprocessed data
data = np.load("splits.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

n_features = X_train.shape[1]
print(f"Input features: {n_features}")

# Build ANN
model = Sequential([
    Dense(64, activation="relu", input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(16, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")   # binary output -> probability of heart disease
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=2
)

# Evaluate on test set
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model + history
model.save("cardiopredict_ann.keras")
joblib.dump(history.history, "training_history.pkl")

print("\n✅ Model trained and saved as cardiopredict_ann.keras")