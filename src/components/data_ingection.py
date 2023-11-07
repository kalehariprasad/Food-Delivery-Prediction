
from src.config.configuration import *
from src.components.data_transformation import DataTransformation,featureengineering
from src.components.model_trainer import ModelTrainer
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomeExeption
from dataclasses import dataclass


logging.info("creating a CLASS call DataIngectionConfig")
@dataclass
class DataIngectionConfig:
    train_data_path:str=TRAIN_FILE_PATH
    test_data_path:str=TEST_FILE_PATH
    raw_data_path:str=RAW_DATA_FILE_PATH

class DataIngection:
    def __init__(self):
        self.data_ingection_config=DataIngectionConfig()


    def Intiate_data_ingection(self):

        logging.info("started data injection ")
        try:
            df=pd.read_csv(DATASET_PATH)
            #df=pd.read_csv(os.path.join("data\finalTrain.csv"))
            os.makedirs(os.path.dirname(self.data_ingection_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingection_config.raw_data_path,index=False)
            logging.info("reding data frame compleeted")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            os.makedirs(os.path.dirname(self.data_ingection_config.train_data_path),exist_ok=True,)
            train_set.to_csv(self.data_ingection_config.train_data_path,header=True)
            logging.info("reading train data set complteeted")
            os.makedirs(os.path.dirname(self.data_ingection_config.test_data_path),exist_ok=True)
            test_set.to_csv(self.data_ingection_config.test_data_path,header=True)
            logging.info("reading test data set complteeted")
            
            return (
                  train_set,test_set
                  )

        except Exception as e:
            raise CustomeExeption(e,sys)
        
if __name__ =="__main__":
    obj=DataIngection()
    train_path, test_path=obj.Intiate_data_ingection()
    data_transformation=DataTransformation()
    train_arry, test_arry,= data_transformation.intiate_data_transformation(train_path, test_path)
    model_trainer=ModelTrainer()
    model_trainer.intiate_model_training(train_arry, test_arry)