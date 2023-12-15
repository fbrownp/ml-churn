from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    STATUS_FILE_DTYPE: str
    unzip_data_dir: Path 
    all_schema: dict    


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transformation_path: Path


@dataclass(frozen=True)
class DataClusteringConfig:
    root_dir: Path
    train_data_path: Path
    model_name: str
    n_clustering: int
    target_column: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    model_name_1: str
    params: dict
    target_column: float

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    model_name: str
    target_column: str 
    mlflow_uri: str
    all_params: dict
