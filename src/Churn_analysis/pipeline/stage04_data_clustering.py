from Churn_analysis.config.configuration import ConfigurationManager
from Churn_analysis.components.data_clustering import DataClustering
from Churn_analysis import logger
import sys
from Churn_analysis.exception.exception import CustomException

STAGE_NAME = "Data Clustering Stage"

class DataClusteringPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_clustering_config = config.get_data_clustering_config()
        data_clustering = DataClustering(config = data_clustering_config)
        data_clustering.get_data_clustering_object()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataClusteringPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)
