import os 
from Churn_analysis import logger
import pandas as pd
import joblib
from kmodes.kprototypes import KPrototypes
from Churn_analysis.entity.config_entity import DataClusteringConfig

class DataClustering():
    def __init__(self, config: DataClusteringConfig):
        self.config = config

    def get_data_clustering_object(self):

        """
        This function is responsible clustering the data
        by KPrototypes methodd
        """

        # Read data
        train_data_df      = pd.read_csv(self.config.train_data_path)

        # Drop customerID
        train_data_df.drop(columns = "customerID", inplace= True)

        # Params of Kprototype
        params_clustering  =  {"n_clusters": self.config.n_clustering}
    
        # Initialize the model and .fit()
        kP = KPrototypes(**params_clustering, init='Huang', n_init=1, verbose=True)
        clustering_model = kP.fit(train_data_df.drop(columns="Churn"),
                                  categorical=[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16])
        
        # Predict groups and create
        cluster_predicted_data = pd.DataFrame(kP.predict(train_data_df.drop(columns = "Churn"),
                                                         categorical=[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16]),
                                                         columns=["Groups"])
        

        clustered_data = pd.concat([train_data_df , cluster_predicted_data], axis = 1)
        clustered_data.to_csv(os.path.join(self.config.root_dir, "Clustered_data.csv"), index = False)


        joblib.dump(clustering_model, os.path.join(self.config.root_dir,self.config.model_name))

        logger.info(f"Clustering model is saved in {os.path.join(self.config.root_dir,self.config.model_name)}")
        logger.info(f"Clustered data is saved in {os.path.join(self.config.root_dir,'Clustered_data')}")


       