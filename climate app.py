import streamlit as st
import pandas as pd
import pickle

# ==============================
# Page Setup
# ==============================
st.set_page_config(page_title="Climate & Accident Prediction", page_icon="🚗", layout="centered")

# Title & Subtitle
st.markdown("<h1 style='text-align:center; color:#1a73e8;'>🚗 Climate & Road Accident Impact Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Predict road accident risk based on environmental & driving conditions</p>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    model = pickle.load(open("best_climate_accident_model.pkl", "rb"))
    return model

try:
    best_model = load_model()
except:
    st.error("❌ Model file not found! Please upload 'best_climate_accident_model.pkl' to the same folder.")
    st.stop()

# ==============================
# Input Form
# ==============================
st.subheader("📥 Enter Scenario Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Driver_Age = st.number_input("Driver Age", 16, 90, 30)
        Driver_Experience = st.number_input("Driver Experience (Years)", 0, 50, 5)
        Number_of_Vehicles = st.number_input("Number of Vehicles", 1, 10, 1)
        Speed_Limit = st.number_input("Speed Limit", 10, 200, 60)
        Driver_Alcohol = st.selectbox("Driver Alcohol", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        Traffic_Density = st.selectbox("Traffic Density", [1,2,3], format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])

    with col2:
        Time_of_Day = st.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night"])
        Road_Type = st.selectbox("Road Type", ["Highway","Urban","Rural"])
        Vehicle_Type = st.selectbox("Vehicle Type", ["Car","Bike","Truck","Bus","Other"])
        Road_Condition = st.selectbox("Road Condition", ["Dry","Wet","Snow","Gravel"])
        Road_Light_Condition = st.selectbox("Light Condition", ["Day","Night","Low Light"])
        Accident_Severity = st.selectbox("Accident Severity", ["Low","Medium","High"])
        Weather = st.selectbox("Weather", ["Clear","Rain","Fog","Snow","Storm"])

    submitted = st.form_submit_button("🔮 Predict")

# ==============================
# Prediction function
# ==============================
if submitted:
    input_data = pd.DataFrame([{
        "Driver_Age": Driver_Age,
        "Time_of_Day": Time_of_Day,
        "Driver_Experience": Driver_Experience,
        "Number_of_Vehicles": Number_of_Vehicles,
        "Traffic_Density": Traffic_Density,
        "Road_Type": Road_Type,
        "Vehicle_Type": Vehicle_Type,
        "Driver_Alcohol": Driver_Alcohol,
        "Speed_Limit": Speed_Limit,
        "Road_Condition": Road_Condition,
        "Road_Light_Condition": Road_Light_Condition,
        "Accident_Severity": Accident_Severity,
        "Weather": Weather
    }])

    prediction = best_model.predict(input_data)[0]

    st.success(f"### 🚨 Predicted Accident Impact Score: **{prediction:.2f}**")
    st.json(input_data.to_dict(orient="records")[0])

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>Real-time prediction powered by Machine Learning</p>", unsafe_allow_html=True)
