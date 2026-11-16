"""
app.py
------
Streamlit web app: Customer Support Ticket Classifier.

Run from the project root:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from predict import MODEL_PATH, load_model
from preprocessing import clean_text
from responses import get_auto_reply

LOW_CONFIDENCE_LIMIT = 40  # percent

st.set_page_config(page_title="Customer Support Ticket Classifier", page_icon="🎫")


@st.cache_resource
def get_model():
    """Load the model once and reuse it (faster than loading on every click)."""
    return load_model()


# ---------- Header ----------
st.title("🎫 Customer Support Ticket Classifier")
st.write(
    "This app reads a customer support ticket and predicts its category "
    "(Login Issue, Application Error, Report, Account Update or Performance) "
    "using TF-IDF and Logistic Regression."
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("Built as an AI/ML fresher project with Python and Scikit-learn.")
    st.write("Trained on a small synthetic dataset, so treat predictions as a demo.")
    st.header("Try an example")
    examples = {
        "Login": "I cannot sign in, my password is rejected.",
        "Error": "The app crashes when I click the save button.",
        "Report": "I need a monthly sales report in Excel.",
        "Account": "Please update my email address on my profile.",
        "Slow": "The website is very slow and keeps lagging.",
    }
    for label, example_text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["ticket_text"] = example_text

# ---------- Model check ----------
if not Path(MODEL_PATH).exists():
    st.error("Trained model not found. Please run `python src/train.py` first.")
    st.stop()

model = get_model()

# ---------- Input ----------
ticket_text = st.text_area(
    "Enter the ticket description:",
    key="ticket_text",
    height=150,
    placeholder="Example: I forgot my password and the reset email is not coming.",
)

if st.button("Predict Category", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket description first.")
    else:
        probabilities = model.predict_proba([clean_text(ticket_text)])[0]
        best_index = probabilities.argmax()
        category = model.classes_[best_index]
        confidence = probabilities[best_index] * 100

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("Predicted category", category)
        col2.metric("Confidence", f"{confidence:.1f}%")

        if confidence < LOW_CONFIDENCE_LIMIT:
            st.warning("Low confidence. A human agent should review this ticket.")

        st.subheader("Probability for each category")
        chart_data = pd.Series(probabilities * 100, index=model.classes_)
        st.bar_chart(chart_data)

        st.subheader("Suggested auto-reply")
        st.info(get_auto_reply(category))
