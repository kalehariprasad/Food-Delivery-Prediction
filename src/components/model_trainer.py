import os,sys
from src.constants import * # importing all variables from constant file
from src.config.configuration import * # importing all from 
from src.logger import logging
from src.exception import CustomeExeption
from src.components.data_transformation import DataTransformation
from src.utils import save_obj,model_evaluation
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ModelTrainerConfig:
    trained_mode_path=MODEL_FILE_PATH

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def intiate_model_training(self,train_arry,test_arry):
        try:
            X_train,y_train,X_test,y_test=(train_arry[:,:-1],train_arry[:,-1],test_arry[:,:-1],test_arry[:,-1])
            models={
                "SVR":SVR(),
                "RandomForestRegressor":RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor()


            }
            model_report:dict=model_evaluation(X_train,y_train,X_test,y_test,models)
            print(model_report)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values().index(best_model_score))

            ]
            best_model=models[best_model_name]
            logging.info(f"best model for the given model is {best_model}with r2_score of {best_model_score}")
            save_obj(file_path=self.model_trainer_config.trained_mode_path,obj=best_model)

        except Exception as e:
            raise CustomeExeption(e,sys)