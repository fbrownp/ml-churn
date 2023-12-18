import requests
import pandas as pd 

class shap_values:
    def __init__(self,values,base,data):
        self.values = values
        self.base_values= base
        self.data = data


var = {
  "customerID": "1",
  "gender": "Male",
  "SeniorCitizen": "Yes",
  "Partner": "No",
  "Dependents": "No",
  "tenure": 20,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Mailed check",
  "MonthlyCharges": 204,
  "TotalCharges": 804,
  "Churn": "Yes"
}

r = requests.post('https://ml-churn-ojc4v2wzja-uc.a.run.app/predict', json=var).json()

shap_content = shap_values(r["Shap__values"], r["Shap__base_values"], r["Shap__data"])


request_as_pandas = r
print(r)
