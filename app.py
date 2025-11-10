import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("house_price_model.pkl")

st.title("🏡 Indian House Price Predictor")

# Input fields
bedrooms = st.number_input("Number of bedrooms", 1, 10)
bathrooms = st.number_input("Number of bathrooms", 1, 10)
area = st.number_input("Living area (sqft)", 500, 10000)
floors = st.number_input("Number of floors", 1, 4)
condition = st.number_input("Condition of the house (1–5)", 1, 5)
grade = st.number_input("Grade of the house (1–10)", 1, 10)

if st.button("Predict Price 💰"):
    input_data = pd.DataFrame([[bedrooms, bathrooms, area, floors, condition, grade]],
        columns=["number of bedrooms", "number of bathrooms", "living area", 
                 "number of floors", "condition of the house", "grade of the house"])
    price = model.predict(input_data)[0]
    st.success(f"🏠 Predicted House Price: ₹{price:,.2f}")

