import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Chargement des fichiers sauvegardés
model = joblib.load('modele_diabete.pkl')
scaler = joblib.load('scaler_diabete.pkl')

# config page
st.set_page_config(page_title="Predict Diabete", layout="centered")
st.title("Système Intelligent de Prédiction du Risque de Diabète")
st.write("Cet systme utilise le modèle XGBoost pour évaluer le risque de diabète.")

# Formulaire patient
st.subheader("Informations du Patient")
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Grossesses", 0, 20, 1)
    gluco = st.number_input("Glucose (mg/dL)", 0, 300, 120)
    bp = st.number_input("Pression Artérielle", 0, 150, 70)
    skin = st.number_input("Épaisseur du pli cutané", 0, 100, 20)

with col2:
    insu = st.number_input("Insuline", 0, 900, 80)
    bmi = st.number_input("IMC (BMI)", 0.0, 70.0, 25.0)
    dpf = st.number_input("Fonction Pedigree Diabète", 0.0, 3.0, 0.5)
    age = st.number_input("Âge", 1, 120, 30)

# Logique predict
if st.button("Analyser le risque"):
    # Tableau avec les données saisies
    features = np.array([[preg, gluco, bp, skin, insu, bmi, dpf, age]])
    
    # scaler
    features_scaled = scaler.transform(features)
    
    # Prédict
    prediction = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled)[0][1] 

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"⚠️ Risque Élevé détecté (Probabilité : {proba:.2%})")
        st.write("Le modèle suggère que ce patient présente des marqueurs de diabète.")
    else:
        st.success(f"✅ Risque Faible (Probabilité : {proba:.2%})")
        st.write("Le modèle suggère que ce patient est sain.")