import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
st.set_page_config(page_title="Smart Housing Efficiency", page_icon="🏠", layout="centered")

st.title("Smart Housing Efficiency Prediction")
st.write(
    "Explore insights into smart home device usage with this interactive tool. This app captures detailed metrics on various device types including lights, security systems, smart speakers, thermostats, and cameras. Features such as usage patterns, energy consumption, malfunction incidents, and user preferences are analyzed to predict device efficiency."
)
st.header("Feature Descriptions")
st.write("-" * 100)

descriptions = {
    "UsageHoursPerDay": "Average hours per day the device is used.",
    "EnergyConsumption": "Daily energy consumption of the device (kWh).",
    "UserPreferences": "User preference for device usage (0 - Low, 1 - High).",
    "MalfunctionIncidents": "Number of malfunction incidents reported.",
    "DeviceAgeMonths": "Age of the device in months.",
}

device_mapping = {
    "CAMERA": "0000",
    "LIGHTS": "1000",
    "SECURITY SYSTEM": "0100",
    "SMART SPEAKERS": "0010",
    "THERMOSTAT": "0001",
}

for feature, description in descriptions.items():
    st.markdown(f"**{feature}:** {description}")

st.write("-" * 100)
st.write(
    "Availaible Devices: Lights, Security System, Smart-Speakers, Thermostat, Camera."
)
st.write("-" * 100)
st.write(
    "Please enter these Encoded Values for the first 4 features to get the Prediction."
)
for feature, description in device_mapping.items():
    st.markdown(f"**{feature}:** {description}")
st.write("-" * 100)
# Using st.columns() to create a grid layout
col1, col2 = st.columns(2)

# Input fields arranged side by side
with col1:
    feature1 = int(st.number_input("Feature 1", key="feature1", step=None, value=0))
    feature2 = int(st.number_input("Feature 2", key="feature2", step=None, value=0))
    feature3 = int(st.number_input("Feature 3", key="feature3", step=None, value=0))
    feature4 = int(st.number_input("Feature 4", key="feature4", step=None, value=0))

with col2:
    feature5 = float(st.number_input("Usage Hours Per Day", key="feature5", step=None, value=0.0))
    feature6 = float(
        st.number_input(
            "Energy Consumption (KWh)", key="feature6", step=None, value=0.0
        )
    )
    feature7 = st.number_input(
        "User Preferences (0 - Low, 1 - High)",
        key="feature7",
        min_value=0,
        max_value=1,
        step=None,
        value=0
    )
    feature8 = st.number_input(
        "Malfunction Incidents", key="feature8", step=None, value=0
    )
    feature9 = st.number_input(
        "Device Age in Months", key="feature9", step=None, value=0
    )

pred_mapping = {0: "Non Efficient", 1: "Efficient"}

if st.button("Predict"):
    # Create the input array
    input_features = np.array(
        [
            [
                feature1,
                feature2,
                feature3,
                feature4,
                feature5,
                feature6,
                feature7,
                feature8,
                feature9,
            ]
        ]
    )

    # Scale the input features
    input_features_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_features_scaled)

    # Display prediction
    if prediction[0] == 1:
        st.write(f"Hey, Your device is predicted to be: {pred_mapping[prediction[0]]}")
    else:
        st.write(f"Hey, Your device is predicted to be: {pred_mapping[prediction[0]]}\nYou can improve the efficiency by following these steps:\n1. Reduce the usage hours per day.\n2. Reduce the energy consumption.\n3. Set the user preferences to low.\n4. Report any malfunction incidents.\n5. Replace the device with a new one.")
