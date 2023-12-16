from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np



class VariablesIn(BaseModel):
    customerID: object
    gender: object
    SeniorCitizen: object
    Partner: object
    Dependents: object
    tenure: int
    PhoneService: object
    MultipleLines: object
    InternetService: object
    OnlineSecurity: object
    OnlineBackup: object
    DeviceProtection: object
    TechSupport: object
    StreamingTV: object
    StreamingMovies: object
    Contract  : object
    PaperlessBilling: object
    PaymentMethod: object
    MonthlyCharges: float
    TotalCharges: float
    Churn: object


class NumberResponse:
    def __init__(self, value: float):
        self.value = value


app = FastAPI()
@app.get("/")
def home():
    return {"Hello world"}

@app.post("/predict")
def predict(data : VariablesIn):
    data_df = pd.DataFrame([dict(data)])
    model = joblib.load("artifacts/model_trainer/churn_predictive_model.joblib")

    input_encoder = joblib.load("artifacts/data_transformation/transformation.pkl")

    Encoded_data = input_encoder.transform(data_df.drop(columns=["customerID"]))
    Encoded_df = pd.DataFrame(Encoded_data , columns= input_encoder.get_feature_names_out())

    Encoded_df.drop(columns= ["Onehot__OnlineSecurity_No internet service",
                                "Onehot__OnlineBackup_No internet service",
                                "Onehot__DeviceProtection_No internet service",
                                "Onehot__TechSupport_No internet service",
                                "Onehot__StreamingTV_No internet service",
                                "Onehot__StreamingMovies_No internet service",
                                "Ordinal__Churn"], inplace= True)
    

    return {"Churn" : float(model.predict(Encoded_df)) }

if __name__ == "__main__":
    uvicorn.run(app , host="127.0.0.1", port=8000)

