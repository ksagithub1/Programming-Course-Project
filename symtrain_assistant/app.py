import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)



import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

from symtrain_assistant.config import DATA_PROCESSED
from symtrain_assistant.fewshot_gpt import run_fewshot_pipeline



@st.cache_data
def load_dataset():
    df = pd.read_parquet(DATA_PROCESSED / "simulations_labeled.parquet")
    return df


@st.cache_resource
def load_knn():
    knn = joblib.load(DATA_PROCESSED / "category_knn.joblib")
    return knn


def main():
    st.title("SymTrain Simulation Intelligence Assistant")
    st.write(
        "Enter a customer request. The system will predict the category, reason, "
        "and steps using a few-shot GPT pipeline."
    )

    df = load_dataset()
    knn = load_knn()

    user_input = st.text_area(
        "Customer request",
        value="Hi, I ordered a shirt last week and need to update the payment method.",
        height=150,
    )

    if st.button("Generate steps"):
        with st.spinner("Thinking..."):
            result = run_fewshot_pipeline(user_input, df, knn)

        st.subheader("Structured Output")
        st.json(result)

        st.subheader("Readable Steps")
        st.write(f"**Category:** {result.get('category', '')}")
        st.write(f"**Reason:** {result.get('reason', '')}")

        steps = result.get("steps", [])
        for i, step in enumerate(steps, start=1):
            st.write(f"{i}. {step}")

    st.sidebar.header("About")
    st.sidebar.write("Built for the SymTrain Simulation Intelligence Assistant project.")


if __name__ == "__main__":
    main()
