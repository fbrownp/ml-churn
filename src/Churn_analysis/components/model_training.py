import os 
from Churn_analysis import logger
from xgboost import XGBClassifier
import joblib
import pandas as pd
from Churn_analysis.entity.config_entity import ModelTrainerConfig


class ModelTrainer():
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_model_trainer_object(self):

        """
        This function is responsible for training the model
        """
        train_data_df     = pd.read_csv(self.config.train_data_path)
        
        y_data_train_1 = train_data_df[self.config.target_column]
        X_data_train_1 = train_data_df.drop(columns= self.config.target_column)

        params_xgb  =  self.config.params
    
        xgb_churn = XGBClassifier(**params_xgb,objective='binary:logistic', random_state=42)
        xgb_churn.fit(X_data_train_1,y_data_train_1)


        joblib.dump(xgb_churn, os.path.join(self.config.root_dir,self.config.model_name_1))

        logger.info(f"{self.config.model_name_1} is saved in {os.path.join(self.config.root_dir,self.config.model_name_1)}")

