import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import streamlit as st
import pickle
import sklearn
import numpy as np

colgpa = pd.read_csv("promedio.csv")

st.title("Promedio")

tab1, tab2, tab3 = st.tabs(["Tab1", "Tab2", "Tab3"])

with tab1:
    st.header("Analisis univariado")
    #Analisis univariado
    fig, ax = plt.subplots(1,4, figsize=(10,4))
    #Alcohol
    ax[0].hist(colgpa["Alcohol"])
    #Trabajo20
    conteo = colgpa["Trabajo20"].value_counts()
    ax[1].bar(conteo.index, conteo.values)
    #Pareja
    conteo1 = colgpa["Pareja"].value_counts()
    ax[2].bar(conteo1.index, conteo1.values)
    #Promedio
    ax[3].hist(colgpa["Promedio"])
    
    st.pyplot(fig)

with tab2:
    st.header("Analisis bivariado") 
    fig, ax = plt.subplots(1,3, figsize=(10,4))
    sns.scatterplot(data=colgpa, x="Alcohol", y="Promedio", ax=ax[0])
    sns.boxplot(data=colgpa, x="Trabajo20", y="Promedio", ax=ax[1])
    sns.boxplot(data=colgpa, x="Pareja", y="Promedio", ax=ax[2])

    st.pyplot(fig)

with open("model.pickle", "rb") as f:
    modelo = pickle.load(f)
with tab3:
    alcohol= st.slider("Alcohol", 0,20)
    job20 = st.selectbox("Trabaja mas de 20 horas", ["Si", "No"])
    if job20 == "Si":
        job20 = 1
    else:
        job20 = 0
    bgfriend = st.selectbox("Tiene pareja", ["Si", "No"])
    if bgfriend == "Si":
        bgfriend = 1
    else:
        bgfriend = 0
    if st.button("Predecir"):
        pred = modelo.predict(np.array([[alcohol, job20, bgfriend]]))
        st.write(f"Su promedio sería {round(pred[0],1)}")
    

