import os,sys
from src.constants import * # importing all variables from constant file
from src.config.configuration import * # importing all from 
from src.logger import logging
from src.exception import CustomeExeption
from src.components.data_transformation import DataTransformation,featureengineering
from src.components.model_trainer import ModelTrainer
from src.components.data_ingection import DataIngection
import streamlit as st

class Train:
    def __init__(self):
        self.c=0
        print(f'********************{self.c}*****************')
    def main(self):
        obj=DataIngection()
        train_path, test_path=obj.Intiate_data_ingection()
        feature_engineering=featureengineering()
        data_transformation_obj=DataTransformation()
        train_arry, test_arry, processed_obj_file_path = data_transformation_obj.intiate_data_transformation(train_path, test_path)
        model_trainer=ModelTrainer()
        r2_score_value=model_trainer.intiate_model_training(train_arry, test_arry)
        logging.info(f"model training compleete in traininig pipeline with r2 score of {r2_score_value}")
        st.write(f'mode has trained wit r2 score of {r2_score_value }') 
      
    