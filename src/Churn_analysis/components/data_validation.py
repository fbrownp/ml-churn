import os 
from Churn_analysis import logger
import pandas as pd
from Churn_analysis.entity.config_entity import DataValidationConfig

class DataValidation():
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            if all_cols != list(all_schema):
                validation_status = False
                with open(self.config.STATUS_FILE, "w") as f:
                    f.write(f"Validation status: {validation_status}")
                    logger.info(f"Validation status of columns names: {validation_status}")

            else:
                validation_status =  True
                with open(self.config.STATUS_FILE, "w") as f:
                    f.write(f"Validation status: {validation_status}")
                    logger.info(f"Validation status of columns names: {validation_status}")


        except Exception as e:
            raise e
        

    def validate_all_columns_dtypes(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.dtypes)
            all_schema = self.config.all_schema.values()


            if all_cols != list(all_schema):
                validation_status = False
                with open(self.config.STATUS_FILE_DTYPE, "w") as f:
                    f.write(f"Validation status: {validation_status}")
                    logger.info(f"Validation status of columns dtypes: {validation_status}")
            else:
                validation_status =  True
                with open(self.config.STATUS_FILE_DTYPE, "w") as f:
                    f.write(f"Validation status: {validation_status}")
                    logger.info(f"Validation status of columns dtypes: {validation_status}")
        except Exception as e:
            raise e