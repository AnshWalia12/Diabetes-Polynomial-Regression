import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺"
)

model = joblib.load(
    "model.pkl"
)

poly = joblib.load(
    "poly.pkl"
)

st.title(
    "🩺 Diabetes Prediction"
)

st.write(
    "Polynomial Regression on Diabetes Dataset"
)

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0
    )

    glucose = st.number_input(
        "Glucose"
    )

    bp = st.number_input(
        "Blood Pressure"
    )

    skin = st.number_input(
        "Skin Thickness"
    )

with col2:

    insulin = st.number_input(
        "Insulin"
    )

    bmi = st.number_input(
        "BMI"
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function"
    )

    age = st.number_input(
        "Age"
    )

if st.button(
    "Predict"
):

    data = np.array(
        [[
            pregnancies,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]]
    )

    transformed = poly.transform(
        data
    )

    pred = model.predict(
        transformed
    )[0]

    if pred > 0.5:
        st.error(
            "High chance of Diabetes"
        )
    else:
        st.success(
            "Low chance of Diabetes"
        )

    st.write(
        "Prediction score:",
        round(pred,3)
    )

df = pd.read_csv(
    "diabetes.csv"
)

st.subheader(
    "Dataset Preview"
)

st.dataframe(
    df.head()
)