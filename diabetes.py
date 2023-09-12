# IMPORTING NECESSARY LIBRARIES
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# SETTING PAGE CONFIGURATIONS
st.set_page_config(page_title="Diabetes Check",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )
st.markdown("<h1 style='text-align: center; color: #900C3F ;'>Diabetes Check</h1>", unsafe_allow_html=True)
# CREATING TABS
tab1, tab2 = st.tabs(["Home","Upload and Check"])
# Home menu
with tab1:
    st.markdown("## **TECHNOLOGIES USED :** Python, Streamlit, ML Algorithm")
    st.markdown("## **OVERVIEW :** In this streamlit web app, you can upload your necessary details regarding your health conditions")
    st.markdown("## • You can check whether you have diabetes or not.")
    st.markdown("## • You can also check how accurate the prediction is done.")
# Upload and Check
with tab2:
    st.markdown("### Upload your details below")
    # GETTING DETAILS FROM THE USER
    gender = st.text_input('Choose your gender [0 for female,1 for male,2 for others]', (0,1,2))
    age = st.text_input('Enter you age')
    hypertension = st.text_input('Do you have hypertension?[0 for no,1 for yes]',(0,1))
    heart_disease = st.text_input('Do you have any heart disease?[0 for no,1 for yes]',(0,1))
    smoking_history = st.text_input('What is your smoking history?[0 for no info,1 for current,2 for ever,3 for former,4 for never,5 for not current]',(0,1,2,3,4,5))
    bmi = st.text_input('Enter you bmi')
    HbA1c_level	= st.text_input('Enter your HbA1c level')
    blood_glucose_level = st.text_input('Enter your blood glucose level')
    if st.button('submit'):
        df = pd.read_csv('diabetes_prediction_dataset.csv')
        # PREPROCESSING
        enc=OrdinalEncoder()
        df["smoking_history"]=enc.fit_transform(df[["smoking_history"]])
        df["gender"]=enc.fit_transform(df[["gender"]])
        # INDEPENDENT AND DEPENDENT VARIABLES
        x= df.drop("diabetes",axis=1)
        y=df["diabetes"]
        # SPLITTING TRAINING AND TESTING DATA
        x_train , x_test , y_train, y_test = train_test_split(x,y,test_size=0.3)
        # TRAINING THE MODEL
        model = DecisionTreeClassifier().fit(x_train,y_train)
        details = np.array([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]])
        y_pred = model.predict(details)
        st.metric(label='Your result',value=y_pred)
        y_pred1 = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred1)
        percentage = accuracy*100
        st.metric(label='accuracy',value=percentage)
