import streamlit as st
import requests
import pandas as pd
import shap
import joblib
import numpy as np
from streamlit_shap import st_shap



# Replace this URL with the URL where your FastAPI app is running
FASTAPI_URL = "http://127.0.0.1:8000"
FASTAPI_URL_PREDICT = 'http://127.0.0.1:8000/predict'

class shap_values:
    def __init__(self,values,base,data):
        self.values = np.array(values)
        self.base_values= np.array(base)
        self.data = np.array(data)



def Input_order(customerID,gender_value,senior_citizen_value, partner_value, dependents_value, tenure_value,
                       phone_service_value, multiple_lines_value, internet_service_value, online_security_value,
                       online_backup_value, device_protection_value, tech_support_value, streaming_tv_value,
                       streaming_movies_value, contract_value, paperless_billing_value, payment_method_value,
                       monthly_charges_value, total_charges_value, churn):
    output_var = {
    "customerID": customerID,
    "gender": gender_value,
    "SeniorCitizen": senior_citizen_value,
    "Partner": partner_value,
    "Dependents": dependents_value,
    "tenure": tenure_value,
    "PhoneService": phone_service_value,
    "MultipleLines": multiple_lines_value,
    "InternetService": internet_service_value,
    "OnlineSecurity": online_security_value,
    "OnlineBackup": online_backup_value,
    "DeviceProtection": device_protection_value,
    "TechSupport": tech_support_value,
    "StreamingTV": streaming_tv_value,
    "StreamingMovies": streaming_movies_value,
    "Contract": contract_value,
    "PaperlessBilling": paperless_billing_value,
    "PaymentMethod": payment_method_value,
    "MonthlyCharges": monthly_charges_value,
    "TotalCharges": total_charges_value,
    "Churn": churn
    }
    return output_var




def main():
    st.title("App to determine Churn rate of clients")
    st.write("This is something i write")

    # Create a sidebar
    st.sidebar.title("Client variables")



    gender_value = st.sidebar.radio("Gender", ["Male","Female"])
    senior_citizen_value = st.sidebar.radio("Senior Citizen", ["Yes","No"])
    partner_value = st.sidebar.radio("Partner", ["Yes","No"])
    dependents_value = st.sidebar.radio("Dependents", ["Yes","No"])
    tenure_value  = st.sidebar.number_input("Tenure", min_value=0, max_value=300, value=0)
    phone_service_value = st.sidebar.radio("Phone Service", ["Yes","No"])
    multiple_lines_value = st.sidebar.radio("Multiple Lines", ["Yes","No","No internet service"])
    internet_service_value = st.sidebar.radio("Internet Service", ["Fiber optic","DSL","No"])
    online_security_value = st.sidebar.radio("Online Security", ["Yes","No","No internet service"])
    online_backup_value = st.sidebar.radio("Online Backup", ["Yes","No","No internet service"])
    device_protection_value = st.sidebar.radio("Device protection", ["Yes","No","No internet service"])
    tech_support_value = st.sidebar.radio("Tech support", ["Yes","No","No internet service"])
    streaming_tv_value = st.sidebar.radio("Streaming TV", ["Yes","No","No internet service"])
    streaming_movies_value = st.sidebar.radio("Streaming Movies", ["Yes","No","No internet service"])
    contract_value = st.sidebar.radio("Contract", ["Month-to-month","One year", "Two year"])
    paperless_billing_value = st.sidebar.radio("PaperlessBilling", ["Yes","No"])
    payment_method_value = st.sidebar.radio("PaymentMethod", ["Mailed check","Bank transfer (automatic)","Electronic check", "Credit card (automatic)"])
    monthly_charges_value = st.sidebar.number_input("MonthlyCharges", min_value=0, max_value=300, value=0)
    total_charges_value = st.sidebar.number_input("TotalCharges", min_value=0, max_value=300, value=0)

    input_variables = Input_order("1",gender_value,senior_citizen_value, partner_value, dependents_value, tenure_value,
                       phone_service_value, multiple_lines_value, internet_service_value, online_security_value,
                       online_backup_value, device_protection_value, tech_support_value, streaming_tv_value,
                       streaming_movies_value, contract_value, paperless_billing_value, payment_method_value,
                       monthly_charges_value, total_charges_value, "Yes")



    col1,col2 = st.columns(2)



    # Create a button
    if st.sidebar.button("Predict"):
        # Action to execute when the button is clicked
        response = requests.post(FASTAPI_URL_PREDICT, json=input_variables).json()

        churn_rate = float(response["Churn"])
        shap_content = shap_values(response["Shap__values"], response["Shap__base_values"], response["Shap__data"])
        Explainer_obj = shap.Explanation(shap_content.values, shap_content.base_values, shap_content.data,
                                            feature_names=response["Shap__columns"])
        cluster_value = response["cluster_value"]

        with col1:
            st.header(f"Churn probability : {round(churn_rate,2)}")
            st.header(f"Type of client : {cluster_value}")


        with col2:
            st.title("Waterfalll:")
            
            st_shap(shap.plots.waterfall(Explainer_obj, max_display=8),400)




if __name__ == "__main__":
    main()