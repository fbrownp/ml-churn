import requests
import pandas as pd 

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

r = requests.post('http://127.0.0.1:8000/predict', json=var).json()

request_as_pandas = pd.DataFrame([r])
print(request_as_pandas)
