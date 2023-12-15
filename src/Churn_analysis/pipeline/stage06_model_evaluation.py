from Churn_analysis.config.configuration import ConfigurationManager
from Churn_analysis.components.model_evaluation import ModelEvaluation
from Churn_analysis import logger
import sys
from Churn_analysis.exception.exception import CustomException

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config = model_evaluation_config)
        model_evaluation.get_model_evaluation_object()




if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)

