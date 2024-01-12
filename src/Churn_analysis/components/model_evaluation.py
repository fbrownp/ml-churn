import os
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import joblib
import pandas as pd
from Churn_analysis.utils.common import save_json, calculate_metrics
import mlflow
import mlflow.xgboost
from  urllib.parse import urlparse
from pathlib import Path
from Churn_analysis.entity.config_entity import ModelEvaluationConfig


DASGHUB_KEY = os.environ["DAGSHUGKEY"]
try_var_1 = 1

class ModelEvaluation():
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/FBrownp/ml-churn.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"]="FBrownp"
        os.environ["MLFLOW_TRACKING_PASSWORD"]= DASGHUB_KEY


    def get_model_evaluation_object(self):
        
        test_data_df      = pd.read_csv(self.config.test_data_path)
        
        y_test = test_data_df[self.config.target_column]
        X_test = test_data_df.drop(columns= self.config.target_column)

        model = joblib.load(os.path.join(self.config.model_path,self.config.model_name))

        conf_matrix = confusion_matrix(y_test, model.predict(X_test))

        y_proba = model.predict_proba(X_test)[:, 1]
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)



        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme



        with mlflow.start_run():

            scores = calculate_metrics(conf_matrix)
            save_json(path= Path(os.path.join(self.config.root_dir,"scores.json")), data = scores)
  
            mlflow.log_params(self.config.all_params)

            for key in scores.keys():
                mlflow.log_metric(key,scores[key])
                print(key,scores[key])
            mlflow.log_metric("ROC_AUC",roc_auc)

            if tracking_url_type_store != "file":
                mlflow.xgboost.log_model(model, "Churn_model", registered_model_name="Churn_model")
            else:
                mlflow.xgboost.log_model(model, "Churn_model")

