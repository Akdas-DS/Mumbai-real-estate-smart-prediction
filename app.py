import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# PAGE CONFIG (LIGHT MODE)
# -------------------------------------------------
st.set_page_config(
    page_title="Mumbai Real Estate Investment Intelligence System",
    page_icon="🏙️",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("mumbai_real_estate_investment_model.pkl")

model = load_model()

# -------------------------------------------------
# APP TITLE & DESCRIPTION
# -------------------------------------------------
st.title("️ Mumbai Real Estate Investment Intelligence System")

st.markdown(
    """
    This application uses a **machine learning–based valuation model** to estimate  
    **price per square foot** and **total property value** for residential properties in Mumbai.
    
    The predictions are designed to support **data-driven real estate investment decisions**.
    """
)

st.divider()

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.header("Property Details")

area = st.sidebar.number_input("Area (sqft)", min_value=200, max_value=10000, value=900)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2)
balconies = st.sidebar.number_input("Balconies", min_value=0, max_value=5, value=1)
age = st.sidebar.number_input("Property Age (years)", min_value=0, max_value=100, value=5)
total_floors = st.sidebar.number_input("Total Floors in Building", min_value=1, max_value=100, value=15)

latitude = st.sidebar.number_input("Latitude", value=19.0760, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=72.8777, format="%.6f")

city = st.sidebar.selectbox("City", ["Mumbai"])
locality = st.sidebar.text_input("Locality", "Andheri West")
property_type = st.sidebar.selectbox("Property Type", ["Apartment", "Villa", "Independent House"])
furnished = st.sidebar.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"])

# -------------------------------------------------
# PREDICTION INPUT DATAFRAME
# -------------------------------------------------
input_data = pd.DataFrame([{
    "area": area,
    "bedroom_num": bedrooms,
    "bathroom_num": bathrooms,
    "balcony_num": balconies,
    "age": age,
    "total_floors": total_floors,
    "latitude": latitude,
    "longitude": longitude,
    "city": city,
    "locality": locality,
    "property_type": property_type,
    "furnished": furnished
}])

# -------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------
if st.button(" Predict Property Value"):
    predicted_price_sqft = model.predict(input_data)[0]
    estimated_value = predicted_price_sqft * area

    st.subheader(" Valuation Results")

    col1, col2 = st.columns(2)

    col1.metric(
        label="Predicted Price per Sqft",
        value=f"₹ {predicted_price_sqft:,.0f}"
    )

    col2.metric(
        label="Estimated Property Value",
        value=f"₹ {estimated_value:,.0f}"
    )

    st.success("Prediction generated successfully")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption(
    "⚠️ Predictions are estimates based on historical data and market assumptions. "
    "They should not be considered as financial advice."
)
