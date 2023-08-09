from src.constants import * # importing all from constants folder 
import os,sys

ROOT_DIR=ROOT_DIR_KEY
logging.info("joining root dirrctory path to dateset director ")
DATASET_PATH=os.path.join(ROOT_DIR,DATA_DIR,DATA_DIR_KEY)
RAW_DATA_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACTS_DIR_KEY,DATA_INJGECTION_RAW_DIR,CURRENT_TIME_STAMP,
                                RAW_DATA_DIR_KEY)
TRAIN_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACTS_DIR_KEY,
                                DATA_INJGECTION_INGESTED_DATA_DIR_KEY,CURRENT_TIME_STAMP,TRAIN_DATA_DIR_KEY)
logging.info('joing train file path compleeted')
TEST_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACTS_DIR_KEY,
                                DATA_INJGECTION_INGESTED_DATA_DIR_KEY,CURRENT_TIME_STAMP,TEST_DATA_DIR_KEY)
logging.info("joining test file path compleeted")