# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
loaded_model=pickle.load(open('C:/Users/meime/Documents/Inlab Internship (ML)/trained_model.sav','rb'))

input_data=(1,85,66,29,0,24.5,0.351,31)
input_data_np=np.asarray(input_data)
input_data_reshaped=input_data_np.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==1):
  print("Having diabetes")
else:
  print("Not having diabetes")