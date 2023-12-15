import os 
from Churn_analysis import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from Churn_analysis.entity.config_entity import DataTransformationConfig



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
        names = list(data.columns[[6,7,8,9,10,11,12,13,14,15,16,17]])
        
        # Applying get_dummies
        for name in names: 
            data = pd.get_dummies(data, columns=[name], dtype=int)


        # Dropping innecesary columns
        data.drop(columns=["customerID","OnlineSecurity_No internet service","OnlineBackup_No internet service",
                           "DeviceProtection_No internet service","TechSupport_No internet service","StreamingTV_No internet service",
                           "StreamingMovies_No internet service"], inplace=True)
        

        Ord_encoder = LabelEncoder()
        Std_encoder = StandardScaler()

        Ordinal_variables = ["gender", "SeniorCitizen","Partner","Dependents","Churn"]
        Numeric_variables = ["tenure","MonthlyCharges","TotalCharges"]

        # Applying transformations
        data[Ordinal_variables] = data[Ordinal_variables].apply(lambda col: Ord_encoder.fit_transform(col))
        data[Numeric_variables] = Std_encoder.fit_transform(data[Numeric_variables])

        # Splitting the data
        train, test = train_test_split(data,test_size=0.3, random_state=42, stratify=data["Churn"] )
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index= False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index= False)

        strat_test  = test.Churn[test["Churn"]==1].sum()/test.Churn.count()
        strat_train = train.Churn[train["Churn"]==1].sum()/train.Churn.count()
        logger.info(f"Transformed and splitted data, stratification of train data {strat_train}, stratification of test data {strat_test}")
        logger.info(train.shape)
        logger.info(test.shape)