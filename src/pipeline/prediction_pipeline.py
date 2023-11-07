import os,sys
from src.constants import * 
from src.config.configuration import *
from src.exception import CustomeExeption
from src.logger import logging
from dataclasses import dataclass
from src.utils import load_model
from src.pipeline.training_pipeline import Train
import pandas as pd
@dataclass
class predictpipelineConfig:
     feature_engineering=FEATURE_ENGINEERING_OBJECT
     preprocessor_path=PREPROSECCING_OBJ_FILE
     model_path=MODEL_FILE_PATH


class PredictionPipeline:
    def __init__(self):
        self.predictpipelineconfig=predictpipelineConfig()

    def predict(self,features):
        try:
            
            preprocessor=load_model(self.predictpipelineconfig.preprocessor_path)

            model=load_model(self.predictpipelineconfig.model_path)
            data_scaled=preprocessor.transform(features)
            pred =model.predict(data_scaled)

            return pred
        except Exception as e:
            logging.info('error occured in PredictPipeline class predicti function')
            raise CustomeExeption(e,sys)
        
class CustomData:
    
        """
            categorical_columns = ['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encoder = ['Road_traffic_density', 'Weather_conditions']
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance']
            columns after dropping unnecessary columns are Index(['Delivery_person_Age', 'Delivery_person_Ratings', 'Weather_conditions',
            'Road_traffic_density', 'Vehicle_condition', 'Type_of_order',
            'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City',
            'Time_taken (min)', 'distance'],
            dtype='object')
        """
        
        def __init__(self,
                 Delivery_person_Age:int,
                 Delivery_person_Ratings:float,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Vehicle_condition:int,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 multiple_deliveries:int,
                 Festival:str,
                 City:str,
                 distance:float):
            self.Delivery_person_Age=Delivery_person_Age
            self.Delivery_person_Ratings=Delivery_person_Ratings
            self.Weather_conditions=Weather_conditions
            self.Road_traffic_density=Road_traffic_density
            self.Vehicle_condition=Vehicle_condition
            self.Type_of_order=Type_of_order
            self.Type_of_vehicle=Type_of_vehicle
            self.multiple_deliveries=multiple_deliveries
            self.Festival=Festival
            self.City=City
            self.distance=distance

        def get_as_DataFrame(self):
             try:

                customdata_dict = {
                    
                'Delivery_person_Age': [self.Delivery_person_Age],
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Weather_conditions': [self.Weather_conditions],
                'Road_traffic_density': [self.Road_traffic_density],
                'Vehicle_condition': [self.Vehicle_condition],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'Type_of_order': [self.Type_of_order],
                'multiple_deliveries': [self.multiple_deliveries],
                'Festival': [self.Festival],
                'City': [self.City],
                'distance': [self.distance]
                }

                data = pd.DataFrame(customdata_dict)

                return data
             except Exception as e:
                logging.info('error occured in CustomData class get_as_DateFrame function')
                raise CustomeExeption(e,sys)
             