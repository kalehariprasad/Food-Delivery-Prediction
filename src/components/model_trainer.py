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
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_path=MODEL_FILE_PATH

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def intiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("'splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "SVR":SVR(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                
                }
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
           
                },
                "SVR":{"kernel":['linear', 'rbf',],
                       

                }
                
            }
        
            model_report=model_evaluation(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,models=models)


            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(f"best model found among models is {best_model}")

            if best_model_score<0.6:
                raise CustomeExeption("No best model found")
                

            save_obj(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_score_value=r2_score(y_test,predicted)
            logging.info(f"r2_score for the best models is {r2_score_value}")
            print(f"best moedel for the proble is {best_model} with r2_score of {r2_score_value}")


            return r2_score_value
            

        except Exception as e:
            raise CustomeExeption(e,sys)