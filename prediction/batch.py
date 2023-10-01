
from src.config.configuration import *
from src.components.data_transformation import DataTransformation,featureengineering
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomeExeption
import os,sys
import pandas as pd
import numpy as np
from src.utils import load_model
from sklearn.pipeline import Pipeline
import pickle


PREDICTION_FOLDER="batch prediction"
PREDICTION_CSV_FOLDER="batch predicton_csv"
INPUT_FILE="input.csv"
PREDICTION_FILE_CSV="output.csv"
FEATUTRE_ENG_FOLDER="feature engineering"
FEATURE_ENG_FILE="feature_eng.csv"
TRANSFORMED_FILE='transformed.csv'

ROOT_DIR=os.getcwd()
BATCH_PREDICTION_INPUT_FILE=os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV_FOLDER,INPUT_FILE)
BATCH_PREDICTION_OUTPUT_FILE=os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV_FOLDER,PREDICTION_FILE_CSV)
FEATURE_ENGINEERING_FILE=os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATUTRE_ENG_FOLDER,FEATURE_ENG_FILE)
PRE_PROCESSED_FILE=os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATUTRE_ENG_FOLDER,TRANSFORMED_FILE)


@dataclass
class BatchPredictionConfig:
    input_file=BATCH_PREDICTION_INPUT_FILE
    output_file=BATCH_PREDICTION_OUTPUT_FILE
    fe_file=FEATURE_ENGINEERING_FILE
    preproccesd_file=PRE_PROCESSED_FILE



class BatchPrediction:
    def __init__(self):
        self.batchpredictionconfig = BatchPredictionConfig()
    
    def start_batch_prediction(self, input_file):
        try:
            feature_eng_file_path = FEATURE_ENGINEERING_FILE_PATH
            transformer_file_path = PREPROSECCING_OBJ_FILE
            model_file_path = MODEL_FILE_PATH

            # Load the feature engineering pipeline
            with open(feature_eng_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)
            # Load the data transformation pipeline
            with open(transformer_file_path, 'rb') as f:
                processor = pickle.load(f)
            # Load the model
            with open(model_file_path, 'rb') as f:
                model = pickle.load(f)
            
            feature_engineering_pipeline=Pipeline([
                ('feature engineering',feature_pipeline)
            ]
            )
            
            df = pd.read_csv(input_file)
            os.makedirs(os.path.dirname(self.batchpredictionconfig.input_file),exist_ok=True)
            df.to_csv(self.batchpredictionconfig.input_file,index=True)


            # Apply feature engineering
            fe_df = feature_engineering_pipeline.transform(df)
            os.makedirs(os.path.dirname(self.batchpredictionconfig.fe_file),exist_ok=True)
            fe_df.to_csv(self.batchpredictionconfig.fe_file,index=True)
        
            # Drop 'Time_taken (min)' column
            target="Time_taken (min)"
            if target in(fe_df.columns):
                fe_df = fe_df.drop(target, axis=1)
            
            transformed_data = processor.transform(fe_df)
            os.makedirs(os.path.dirname(self.batchpredictionconfig.preproccesd_file),exist_ok=True)
            transformed_data.to_csv(self.batchpredictionconfig.preproccesd_file)
           
  
            predictions = model.predict(transformed_data)

            df_prediction = pd.DataFrame(predictions, columns=['predictions'])
            os.makedirs(os.path.dirname(self.batchpredictionconfig.output_file),exist_ok=True)
            df_prediction.to_csv(self.batchpredictionconfig.output_file,index="prddictions")        
            logging.info('Batch prediction done')
            return df_prediction

        except Exception as e:
            raise CustomeExeption(e, sys)

