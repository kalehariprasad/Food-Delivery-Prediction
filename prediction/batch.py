from src.config.configuration import *
from src.components.data_ingection import DataIngection
from src.components.data_transformation import DataTransformation,featureengineering
from src.components.model_trainer import ModelTrainer
import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomeExeption
from src.utils import load_model
from dataclasses import dataclass

class BatchConfig:
    FE_OBJ_PATH=FEATURE_ENGINEERING_FILE_PATH
    PRE_PROCESSOR_PATH=PREPROSECCING_OBJ_FILE
    MODEL_PATH=MODEL_FILE_PATH
    INPUT_FILE=BATCH_INPUT_FILE
    FE_FILE=BATCH_FE_FILE
    TRANSFORMED_FILE=BATCH_TRANSFORMED_FILE
    OUTPUT_FILE=BATCH_OUTPUT_FILE

class BatchPrediction:
    def __init__(self):
        self.batchprediction_config=BatchConfig()

    def initiate_batch_prediction(self,input_file):

        fe_obj=load_model(file_path=self.batchprediction_config.FE_OBJ_PATH)
        process_obj=load_model(file_path=self.batchprediction_config.PRE_PROCESSOR_PATH)
        model=load_model(file_path=self.batchprediction_config.MODEL_PATH)
        os.makedirs(os.path.dirname(self.batchprediction_config.INPUT_FILE),exist_ok=True)
        input_file.to_csv(self.batchprediction_config.INPUT_FILE)
        fe_data=fe_obj.transform(input_file)
        target_column="Time_taken (min)"

        if target_column in fe_data.columns:
            fe_data.drop(target_column,axis=1,inplace=True)
        
        os.makedirs(os.path.dirname(self.batchprediction_config.FE_FILE),exist_ok=True)
        fe_data.to_csv(self.batchprediction_config.FE_FILE)
        transformed_data=process_obj.transform(fe_data)
        os.makedirs(os.path.dirname(self.batchprediction_config.TRANSFORMED_FILE),exist_ok=True)
        transformed_data_df=pd.DataFrame(transformed_data)
        transformed_data_df.to_csv(self.batchprediction_config.TRANSFORMED_FILE)

        prediction=model.predict(transformed_data)
        fe_data['prediction']=prediction
        pred_csv=fe_data[['Delivery_person_Age', 'Delivery_person_Ratings', 'Weather_conditions',
       'Road_traffic_density', 'Vehicle_condition', 'Type_of_order',
       'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City',
       'distance','prediction']]

        return pred_csv




