# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 01:21:07 2024

@author: meime
"""

import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt

loaded_model=pickle.load(open('C:/Users/meime/Documents/Inlab Internship (ML)/trained_model1.sav','rb'))

def diabetes_prediction(input_data):
    input_data_as_np=np.asarray(input_data)
    input_data_reshaped=input_data_as_np.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==0):
        return "This person is not having diabetes"
    else:
       return "This person is having diabetes"
def main():
    st.title("Diabetes Prediction")
    
    #gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes

    gender = st.text_input("Gender(Male(0)orFemale(1)):")
    age = st.text_input("Age:")
    hypertension = st.text_input("Hypertension level(1)or not (0):")
    heart_disease = st.text_input("Heart_disease(1)or not(0):")
    smoking_history = st.text_input("Smoking history(No info(0),Current(1),Ever(2),Former(3),Never(4),Not current(5)):")
    bmi = st.text_input("BMI range:")
    HbA1c_level = st.text_input("HbA1c level(5.7 to 6.4):")
    blood_glucose_level = st.text_input("Blood glucose level:")
    
    diagnosis=''
    
    if(st.button("Result")):
        diagnosis=diabetes_prediction([gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level])
    st.success(diagnosis) 

if __name__ == '__main__':
    main()
    

    
    