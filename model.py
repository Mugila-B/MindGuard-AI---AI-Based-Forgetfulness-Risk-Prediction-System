
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap

np.random.seed(42)
tf.random.set_seed(42)

FEATURES = [
    "Typing Avg",
    "Error Avg",
    "Screen Avg",
    "Switch Avg",
    "Typing Trend",
    "Error Trend",
    "Switch Trend"
]

def trend(x):
    return x[-1] - np.mean(x[:10])

def build_input(typing, errors, screen, switch):
    return np.array([[
        np.mean(typing),
        np.mean(errors),
        np.mean(screen),
        np.mean(switch),
        trend(typing),
        trend(errors),
        trend(switch)
    ]], dtype=np.float32)

def build_sequence(typing, errors, screen, switch):
    seq = np.array(list(zip(typing, errors, screen, switch)), dtype=np.float32)
    return seq.reshape(1, seq.shape[0], seq.shape[1])

def make_synthetic_data(n=2000, days=30):
    X_tab, X_seq, y = [], [], []

    for _ in range(n):
        base_typing = np.random.uniform(25, 60)
        base_errors = np.random.uniform(0, 5)
        base_screen = np.random.uniform(80, 180)
        base_switch = np.random.uniform(5, 20)

        typing = np.array([base_typing + i * np.random.uniform(-0.8, 0.4) + np.random.normal(0, 2) for i in range(days)])
        errors = np.array([base_errors + i * np.random.uniform(0.0, 0.2) + np.random.normal(0, 0.5) for i in range(days)])
        screen = np.array([base_screen + i * np.random.uniform(-0.2, 0.3) + np.random.normal(0, 3) for i in range(days)])
        switch = np.array([base_switch + i * np.random.uniform(0.0, 0.4) + np.random.normal(0, 1) for i in range(days)])

        tab = np.array([
            np.mean(typing),
            np.mean(errors),
            np.mean(screen),
            np.mean(switch),
            typing[-1] - np.mean(typing[:10]),
            errors[-1] - np.mean(errors[:10]),
            switch[-1] - np.mean(switch[:10]),
        ], dtype=np.float32)

        seq = np.array(list(zip(typing, errors, screen, switch)), dtype=np.float32)

        risk_score = (
            (35 - np.mean(typing)) * 0.04 +
            (np.mean(errors) - 2) * 0.25 +
            (np.mean(switch) - 10) * 0.03 +
            max(0, np.std(screen) - 10) * 0.02 +
            max(0, typing[0] - typing[-1]) * 0.01
        )

        prob = 1 / (1 + np.exp(-risk_score))
        label = 1 if prob > 0.5 else 0

        X_tab.append(tab)
        X_seq.append(seq)
        y.append(label)

    return np.array(X_tab), np.array(X_seq), np.array(y)

X_tab, X_seq, y = make_synthetic_data()

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tab, y)

lstm_model = Sequential([
    Input(shape=(30, 4)),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(16),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.fit(X_seq, y, epochs=8, batch_size=32, verbose=0)

explainer = shap.TreeExplainer(rf)

y_rf = rf.predict(X_tab)
y_lstm = (lstm_model.predict(X_seq, verbose=0).flatten() > 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(y, y_rf),
    "precision": precision_score(y, y_rf, zero_division=0),
    "recall": recall_score(y, y_rf, zero_division=0),
    "f1": f1_score(y, y_rf, zero_division=0),
    "confusion_matrix": confusion_matrix(y, y_rf)
}

def get_metrics():
    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    }, metrics["confusion_matrix"]

def predict(typing, errors, screen, switch):
    rf_input = build_input(typing, errors, screen, switch)
    seq_input = build_sequence(typing, errors, screen, switch)

    rf_prob = rf.predict_proba(rf_input)[0][1]
    lstm_prob = float(lstm_model.predict(seq_input, verbose=0)[0][0])

    prob = float(np.clip(0.6 * rf_prob + 0.4 * lstm_prob, 0, 1))

    shap_values = explainer.shap_values(rf_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[1][0]
    else:
        shap_values = shap_values[0]

    shap_values = np.array(shap_values).flatten()

    reasons = []
    for feature, value in zip(FEATURES, shap_values):
        if value > 0.05:
            reasons.append(f"{feature} increases risk level")

    if np.mean(typing) < 35:
        reasons.append("Low typing speed detected")
    if np.mean(errors) > 4:
        reasons.append("High error frequency detected")
    if np.std(screen) > 20:
        reasons.append("Unstable screen usage pattern")

    if prob < 0.3:
        risk = "LOW"
    elif prob < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return prob, risk, reasons
