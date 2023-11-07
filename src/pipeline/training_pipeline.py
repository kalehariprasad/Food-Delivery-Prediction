  
from src.config.configuration import *
from src.components.data_ingection import DataIngection
from src.components.data_transformation import DataTransformation,featureengineering
from src.components.model_trainer import ModelTrainer
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomeExeption
from dataclasses import dataclass

class Train:
    def __init__(self):
        self.c=0
        print(f'********************{self.c}*****************')
    def main(self):

        obj=DataIngection()
        train_path, test_path=obj.Intiate_data_ingection()
        data_transformation=DataTransformation()
        train_arry, test_arry= data_transformation.intiate_data_transformation(train_path, test_path)
        model_trainer=ModelTrainer()
        model_trainer.intiate_model_training(train_arry, test_arry)    