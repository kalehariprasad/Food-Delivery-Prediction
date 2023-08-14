import os ,sys
from datetime import datetime
from src.logger import logging


def get_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

#data injection  variables
CURRENT_TIME_STAMP=get_time_stamp()

ROOT_DIR_KEY=os.getcwd()
DATA_DIR="data"
DATA_DIR_KEY="finalTrain.csv"

ARTIFACTS_DIR_KEY="Artifact"
DATA_INGETION_KEY="raw_data_dir"
DATA_INJGECTION_RAW_DIR="data_ingection"
DATA_INJGECTION_INGESTED_DATA_DIR_KEY="ingected_dir"
RAW_DATA_DIR_KEY="raw.csv"
TRAIN_DATA_DIR_KEY="train.csv"
TEST_DATA_DIR_KEY="test.csv"
# data feature engineering variables
DATA_TRANSFORMATION_ARTICAT_FOLDER="Data transformation" # we are trying create two folders in Artifacts folders named Data transformation & preprocessor
DATA_PROCESSING_FPLDER="preprocessor"
DATA_PREPEOCESSING_OBJECT="preprocessor.pkl"
FEATURE_ENGINEERING_OBJECT="feature engineering.pkl"
DATA_TRANSFORMATION_FOLDER="Transformation"
TRANSFORMATION_TRAIN_FILE="transformed_train.csv"
TRANSFORMATION_TEST_FILE="transformed_test.csv"
#model trainer variable
MODEL_TRAINER_FOLDER="Model trainer"
MODEL_OBJECT="Model.pkl"