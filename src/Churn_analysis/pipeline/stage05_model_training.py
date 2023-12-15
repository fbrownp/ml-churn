from Churn_analysis.config.configuration import ConfigurationManager
from Churn_analysis.components.model_training import ModelTrainer
from Churn_analysis import logger
import sys
from Churn_analysis.exception.exception import CustomException




STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_trainer_config()
        model_training = ModelTrainer(config = model_training_config)
        model_training.get_model_trainer_object()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)
