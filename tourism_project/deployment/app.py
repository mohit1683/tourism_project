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

age = st.number_input("Age", min_value=1, max_value=100, value=50, step=1)

duration_of_pitch = st.number_input("Duration Of Pitch", min_value=0, max_value=10, value=3, step=1)
number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=0, max_value=500, value=4, step=1)
number_of_followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=5, step=1)

number_of_trips = st.number_input("Number Of Trips", min_value=0, max_value=10, value=1)
number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=1)
monthly_income = st.number_input("MonthlyIncome", min_value=0, max_value=10000000, value=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': type_of_contact,
    'Occupation': occupation,
    'Gender': gender,
    'ProductPitched': product_pitched,
    'MaritalStatus': marital_status,
    'Designation': designation,
    'OwnCar': own_car,
    'Passport': passport,
    'CityTier': city_tier,
    'PreferredPropertyStar': preferred_property_star,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'Age' : age,
    'DurationOfPitch' : duration_of_pitch,
    'NumberOfPersonVisiting' : number_of_person_visiting,
    'NumberOfFollowups' : number_of_followups,
    'NumberOfTrips' : number_of_trips,
    'NumberOfChildrenVisiting' : number_of_children_visiting,
    'MonthlyIncome' : monthly_income
}])

# Predict button
if st.button("Predict Result"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Product Taken= {prediction}")
