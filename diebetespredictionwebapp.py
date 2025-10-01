#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 22:37:45 2025

@author: mohammad shaikh
"""

import numpy as np
import streamlit as st
import pickle
with open('trained_model.sav', 'rb') as f:
    loading_model = pickle.load(f)

def diebetes_prediction(input_data):
    input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # prediction
    prediction = loading_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        print("The person is not diabetic")
    else:
        print("The person is diabetic")
        
        

def main():
    st.title("Diabetes Prediction Web App")

    # Taking user inputs
    Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    Glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
    Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0, format="%.1f")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0, format="%.2f")
    Age = st.number_input("Age", min_value=1, max_value=120, value=25)

    # Prediction button
    if st.button("Predict"):
        # Collect input features into a 2D array (since model expects this shape)
        features = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]

        # Run the model prediction
        prediction = loading_model.predict(features)

        # Show result
        if prediction[0] == 1:
            st.error("⚠️ The person is likely to have Diabetes")
        else:
            st.success("✅ The person is not likely to have Diabetes")


    
if __name__=='__main__':
    main()
    
