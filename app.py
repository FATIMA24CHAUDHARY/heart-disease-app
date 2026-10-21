import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open("heart_svc_model.pkl", "rb"))

sex_opts = ["Male", "Female"]
cp_opts = ["Typical", "Atypical", "NonAnginal", "Asymptomatic"]
exang_opts = ["Yes", "No"]
thal_opts = ["Normal", "Fixed", "Reversible"]

sex_enc = LabelEncoder().fit(sex_opts)
cp_enc = LabelEncoder().fit(cp_opts)
exang_enc = LabelEncoder().fit(exang_opts)
thal_enc = LabelEncoder().fit(thal_opts)

st.title("Heart Disease Prediction System")

sex = st.selectbox("Sex", sex_opts)
cp = st.selectbox("Chest pain type", cp_opts)
exang = st.selectbox("Exercise-induced angina", exang_opts)
thal = st.selectbox("Thallium stress test result", thal_opts)

if st.button("Predict"):
    X = pd.DataFrame({
        "Sex": [sex_enc.transform([sex])[0]],
        "ChestPain": [cp_enc.transform([cp])[0]],
        "ExerciseAngina": [exang_enc.transform([exang])[0]],
        "Thalassemia": [thal_enc.transform([thal])[0]],
    })
    if model.predict(X)[0] == 1:
        st.error("Prediction: HEART DISEASE LIKELY")
    else:
        st.success("Prediction: NO HEART DISEASE LIKELY")