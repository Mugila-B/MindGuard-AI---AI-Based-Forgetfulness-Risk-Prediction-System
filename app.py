
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from model import predict, get_metrics
from utils import risk_ui

st.set_page_config(page_title="MindGuard AI", page_icon="🧠", layout="wide")

def gen(base, slope, noise=3):
    return [base + i * slope + np.random.randint(-noise, noise) for i in range(30)]

typing = gen(45, -0.4)
errors = gen(2, 0.2)
screen = gen(120, 0.1)
switch = gen(10, 0.3)

prob, risk, reasons = predict(typing, errors, screen, switch)
ui = risk_ui(risk)

metrics, cm = get_metrics()

st.markdown(
    """
    <style>
    .card {
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(245,247,255,0.95));
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.06);
    }
    .big {
        font-size: 30px;
        font-weight: 700;
    }
    .muted {
        color: #64748b;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 MindGuard AI - Forgetfulness Detection System")
st.caption("AI-based risk estimation using Random Forest, LSTM, and SHAP")

top1, top2, top3, top4 = st.columns(4)

with top1:
    st.markdown(f"<div class='card'><div class='muted'>Risk Level</div><div class='big'>{ui['emoji']} {ui['label']}</div><div class='muted'>Probability: {prob:.2f}</div></div>", unsafe_allow_html=True)
with top2:
    st.markdown(f"<div class='card'><div class='muted'>Accuracy</div><div class='big'>{metrics['accuracy']:.2%}</div></div>", unsafe_allow_html=True)
with top3:
    st.markdown(f"<div class='card'><div class='muted'>Precision</div><div class='big'>{metrics['precision']:.2%}</div></div>", unsafe_allow_html=True)
with top4:
    st.markdown(f"<div class='card'><div class='muted'>Recall / F1</div><div class='big'>{metrics['recall']:.2%} / {metrics['f1']:.2%}</div></div>", unsafe_allow_html=True)

st.markdown("---")

colA, colB = st.columns([1.1, 1])

with colA:
    st.subheader("📊 Behavioral Trends")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(typing, label="Typing Speed", linewidth=2)
    ax.plot(errors, label="Errors", linewidth=2)
    ax.plot(screen, label="Screen Time", linewidth=2)
    ax.plot(switch, label="App Switching", linewidth=2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    st.pyplot(fig)

with colB:
    st.subheader("🧩 Prediction Summary")
    if risk == "LOW":
        st.success("Stable cognitive pattern detected.")
    elif risk == "MEDIUM":
        st.warning("Some irregular activity detected.")
    else:
        st.error("High-risk behavioral pattern detected.")

    st.progress(min(max(prob, 0), 1))

    st.markdown("### AI Explanation")
    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("No significant risk detected.")

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("🧪 Model Metrics")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with m2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("F1 Score", f"{metrics['f1']:.2%}")

with right:
    st.subheader("🗂 Confusion Matrix")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Low", "High"])
    ax2.set_yticklabels(["Low", "High"])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    st.pyplot(fig2)

st.sidebar.header("📅 30-Day Activity Log")
for i in range(30):
    st.sidebar.write(f"Day {i+1}: T={typing[i]} | E={errors[i]} | S={screen[i]} | Sw={switch[i]}")
