import os 
from Churn_analysis import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from Churn_analysis.entity.config_entity import DataTransformationConfig
from Churn_analysis.utils.common import save_object
from sklearn.compose import ColumnTransformer


class DataTransformation():
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def get_train_test_data(self):
        """
        Function that get the data, apply 
        get_dummies to categorical data, StandardScaler to numeric
        and LabelEncoder for choice data.
        It also drop unnecesary columns
        """
        # Reading files
        data = pd.read_csv(self.config.data_path)



        Ordinal_variables = ["gender", "SeniorCitizen","Partner","Dependents","Churn"]
        Numeric_variables = ["tenure","MonthlyCharges","TotalCharges"]
        One_hot_variables = ["PhoneService","MultipleLines", "InternetService","OnlineSecurity","OnlineBackup",
                            "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling",
                            "PaymentMethod"]

        #------------Complete data transformation-----------------------------------------------------------
        preprocessor = ColumnTransformer(transformers=[
                                        ("Ordinal", OrdinalEncoder(), Ordinal_variables),
                                        ("Standard", StandardScaler(), Numeric_variables),
                                        ("Onehot", OneHotEncoder(), One_hot_variables)
                                        ])

        input_encoder = preprocessor.fit(data.drop(columns=["customerID"]))

        Encoded_data = input_encoder.transform(data.drop(columns=["customerID"]))
        Encoded_df = pd.DataFrame(Encoded_data , columns= preprocessor.get_feature_names_out())

        Encoded_df.drop(columns= ["Onehot__OnlineSecurity_No internet service",
                                  "Onehot__OnlineBackup_No internet service",
                                  "Onehot__DeviceProtection_No internet service",
                                  "Onehot__TechSupport_No internet service",
                                  "Onehot__StreamingTV_No internet service",
                                  "Onehot__StreamingMovies_No internet service"], inplace= True)
        
        # --------------------------------------------------------------------------------------------------
        save_object(self.config.transformation_path, input_encoder)

        # Splitting the data
        train, test = train_test_split(Encoded_df,test_size=0.3, random_state=42, stratify=data["Churn"] )
        
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index= False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index= False)

        strat_test  = test[test["Ordinal__Churn"]==1].sum()/test[test["Ordinal__Churn"]==1].count()
        strat_train = train[train["Ordinal__Churn"]==1].sum()/train[train["Ordinal__Churn"]==1].count()

        logger.info(f"Transformed and splitted data, stratification of train data {strat_train} stratification of test data {strat_test}")
        logger.info(train.shape)
        logger.info(test.shape)