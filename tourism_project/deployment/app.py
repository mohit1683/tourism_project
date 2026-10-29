import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourist_mlop_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Visit with us")
st.write("""
This app will predict whether customer will purchase the new tourist package or not
""")

# User input
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business","Small Business","Salaried"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King","Standard","Super Deluxe"])
marital_status = st.selectbox("MaritalStatus", ["Married", "Divorced", "Single"])
designation = st.selectbox("Designation", ["Executive","AVP", "VP", "Manager","Senior Manager"])
own_car = st.selectbox("Own Car", ["1", "0"])
passport = st.selectbox("Passport", ["1", "0"])
city_tier = st.selectbox("City Tier", ["1", "2","3"])
preferred_property_star = st.selectbox("Preferred Property Star", ["3", "4","5"])
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", ["1","2","3", "4","5"])

age = st.number_input("AGE", min_value=1.0, max_value=100, value=50.0, step=1)

duration_of_pitch = st.number_input("Duration Of Pitch", min_value=0, max_value=10, value=3, step=1)
number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=0, max_value=500, value=4, step=1)
number_of_followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=5, step=1)

number_of_trips = st.number_input("Number Of Trips", min_value=0, max_value=10, value=1)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=1)
number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=1)
monthly_income = st.number_input("MonthlyIncome", min_value=0, max_value=10000000, value=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'type_of_contact': type_of_contact,
    'occupation': occupation,
    'gender': gender,
    'product_pitched': product_pitched,
    'marital_status': marital_status,
    'designation': designation,
    'own_car': own_car,
    'passport': passport,
    'city_tier': city_tier,
    'preferred_property_star': preferred_property_star,
    'pitch_satisfaction_score': pitch_satisfaction_score,
    'age' : age,
    'duration_of_pitch' : duration_of_pitch,
    'number_of_person_visiting' : number_of_person_visiting,
    'number_of_followups' : number_of_followups,
    'number_of_trips' : number_of_trips,
    'number_of_children_visiting' : number_of_children_visiting,
    'monthly_income' : monthly_income
}])

# Predict button
if st.button("Predict Result"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Product Taken **{prediction} ")
