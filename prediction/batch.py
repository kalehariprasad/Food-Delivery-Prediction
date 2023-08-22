
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
PREDICTION_FILE_CSV="batch prediction.csv"
FEATUTRE_ENG_FOLDER="feature engineering"

ROOT_DIR=os.getcwd()
BATCH_PREDICTION=os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV_FOLDER)
FEATURE_ENGINEERING=os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATUTRE_ENG_FOLDER)


class BatchPrediction:
    def __init__(self,input_file_path,model_file_path,transformer_file_path,feature_eng_file_path)->None:
        self.input_file_path=input_file_path,
        self.model_file_path=model_file_path,
        self.transformer_file_path=transformer_file_path,
        self.feature_eng_file_path=feature_eng_file_path
    
    def start_batch_prediction(self):
        try:
            #load the feature engineering pipeline path
            with open(self.feature_eng_file_path)as f:
                feature_pipeline=pickle.load(f)
            #load the data transformation path
            with open(self.transformer_file_path)as f:
                processor=pickle.load(f)
            #load model seperately model file path
            model=load_model(file_path=self.model_file_path)
            #feature engineering pipe line
            feature_engineering_pipeline=Pipeline([
                ('feature engineering',feature_pipeline)
            ]
            )
            df=pd.read_csv(self.inpu_file_path)
            df.to_csv('batch_prediction_input_file.csv')

            #apply feature engineering
            df=feature_engineering_pipeline.transform(df)
            df.to_csv('batch_prediction_feature_engineered_input_file.csv')
            FEATURE_ENGINEERING_PATH=FEATURE_ENGINEERING
            os.makedirs(FEATURE_ENGINEERING_PATH,exist_ok=True)
            file_path=os.path.join(FEATURE_ENGINEERING_PATH,'batch_feature_engineerinig')
            df.to_csv(file_path,index=False)

            #time taken columns
        
            df=df.drop("Time_taken (min)",axis=1)
            df.to_csv('Time_taken_dropped.csv')
            transformed_data=processor.transform(df)
            file_path=os.path.join(FEATURE_ENGINEERING,'processor.csv')
             
            predictions=model.predict(transformed_data)

            df_prediction=pd.DataFrame(predictions,columns=['predictions'])
            BATCH_PREDICTION_PATH=BATCH_PREDICTION
            os.makedirs(BATCH_PREDICTION_PATH,exist_ok=True)
            csv_path=os.path.join(BATCH_PREDICTION_PATH,PREDICTION_FILE_CSV)
            df_prediction.to_csv(csv_path,index=False)
            logging.info('batch prediction done')
            






        except Exception as e:
            raise CustomeExeption(e,sys)    
        
