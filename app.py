from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
import json
from kmodes.kprototypes import KPrototypes
#

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



app = FastAPI()
@app.get("/")
def home():
    return {"Hello world"}

@app.post("/predict")
def predict(data : VariablesIn):
    data_df = pd.DataFrame([dict(data)])
    model = joblib.load("artifacts/model_trainer/churn_predictive_model.pkl")
    model_clustering = joblib.load("artifacts/data_clustering/clustering_model.pkl")
    explainer = joblib.load("research/SHAP/shap_explainer")
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
    
    shap_values = explainer(Encoded_df)
    cluster_value = model_clustering.predict(data_df.drop(columns = ["Churn","customerID"]),
                                             categorical=[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16])

    output_ = {"Churn" : float(model.predict_proba(Encoded_df)[:, 1]),
               "Shap__values": list(shap_values.values[0]),
               "Shap__base_values": list(shap_values.base_values),
               "Shap__data": list(shap_values.data[0]),
               "Shap__columns": list(Encoded_df.columns),
               "cluster_value": float(cluster_value)}

    return output_

if __name__ == "__main__":
    uvicorn.run(app , host="0.0.0.0", port=8080)
    # uvicorn.run(app , host="127.0.0.1", port=8080)

