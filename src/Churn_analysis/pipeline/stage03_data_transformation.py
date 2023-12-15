from Churn_analysis.config.configuration import ConfigurationManager
from Churn_analysis.components.data_transformation import DataTransformation
from Churn_analysis import logger
from pathlib import Path
import sys
from Churn_analysis.exception.exception import CustomException


STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f, open(Path("artifacts/data_validation/status_dtype.txt"), "r") as g:
                status = f.read().split(" ")[-1]
                status_dtype = g.read().split(" ")[-1]
            
                if status == "True" and status_dtype == "True":
                    config = ConfigurationManager()
                    data_transformation_config = config.get_data_transformation_config()
                    data_transformation = DataTransformation(config = data_transformation_config)
                    data_transformation.get_train_test_data()
                else:
                    raise Exception(f"Your schema is not valid, check columns is {status}, check dtype is {status_dtype}")
        except Exception as e:
            raise CustomException(e,sys)





if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)
