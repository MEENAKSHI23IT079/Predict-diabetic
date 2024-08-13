# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:31:50 2024

@author: meime
"""

import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt

loaded_model=pickle.load(open('C:/Users/meime/Documents/Inlab Internship (ML)/trained_model.sav','rb'))

def diabetes_prediction(input_data):

    input_data_np=np.asarray(input_data)
    input_data_reshaped=input_data_np.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]==1):
      return "Having diabetes"
    else:
      return "Not having diabetes"
  
def main():
        st.title("diabetes_prediction_webapp")

        Pregnancies = st.text_input("No of Pregnancies")
        Glucose = st.text_input("Glucose level")
        BloodPressure = st.text_input("BloodPressure value")
        SkinThickness = st.text_input("SkinThickness,")
        Insulin = st.text_input("Insulin level")
        BMI = st.text_input("BMI value")
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value")
        Age = st.text_input("Age")
        
        diagnosis = ''
        
        if(st.button("Prediction_result")):
            diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        st.success(diagnosis)
        
        if Glucose:
        # Convert to float for plotting
             glucose_value = float(Glucose)
        
        # Create a histogram
             plt.figure(figsize=(10, 10))
             plt.hist(glucose_value, bins=10)
             plt.title("Histogram of Glucose Level")
             plt.xlabel("Glucose")
             plt.ylabel("Frequency")
             st.pyplot(plt) 
        
if __name__ == '__main__' :
    main()       

           