import os,sys
from src.constants import * 
from src.config.configuration import *
from src.exception import CustomeExeption
from src.logger import logging
from dataclasses import dataclass
from src.utils import load_model
from src.pipeline.training_pipeline import Train
import pandas as pd
import streamlit as st

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
class Streamlit:
    def __init__(self):
       self.predictpipelineconfig=predictpipelineConfig()

    def train(self):
        st.title('Zomato Delivery Prediction training stage')
        train=Train()
        train.main()
     

    def batchprediction(self):
        st.title("batch  Delivery prediction ")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            preprocessor=load_model(self.predictpipelineconfig.preprocessor_path)
            model=load_model(self.predictpipelineconfig.model_path)
            
            scaled_df=preprocessor.transform(df)
            predictions=model.predict(scaled_df)
            st.write(f'predictions are :{predictions}')


    def single_prediction(self):
            st.title(" single Delivery Prediction")
               

            delivery_person_age = st.number_input("Delivery Person Age:", min_value=0, value=25, step=1)
            Delivery_person_Ratings = st.number_input("Delivery Person Ratings:", min_value=0.0, max_value=5.0, value=4.0, step=0.1)

            weather_conditions_options = ['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny']
            Weather_conditions = st.selectbox("Weather Conditions:", sorted(weather_conditions_options))

            road_traffic_density_options = ['Jam', 'High', 'Medium', 'Low']
            road_traffic_density = st.selectbox("Road Traffic Density:", sorted(road_traffic_density_options))

            vehicle_condition = st.number_input("Vehicle Condition:", min_value=0, value=1, step=1)

            type_of_order_options = ['Snack', 'Meal', 'Drinks', 'Buffet']
            type_of_order = st.selectbox("Type of Order:", sorted(type_of_order_options))

            type_of_vehicle_options = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']
            type_of_vehicle = st.selectbox("Type of Vehicle:", sorted(type_of_vehicle_options))

            multiple_deliveries = st.number_input("Multiple Deliveries:", min_value=0, value=0, step=1)

            festival_options = ['No', 'Yes']
            festival = st.selectbox("Festival:", festival_options)

            city_options = ['Metropolitian', 'Urban', 'Semi-Urban']
            city = st.selectbox("City:", sorted(city_options))

            distance = st.number_input("Distance:", min_value=0.0, value=2.0, step=0.1)
            custom_data = CustomData(
        Delivery_person_Age=delivery_person_age,
        Delivery_person_Ratings=Delivery_person_Ratings,
        Weather_conditions=Weather_conditions,
        Road_traffic_density=road_traffic_density,
        Vehicle_condition=vehicle_condition,
        Type_of_order=type_of_order,
        Type_of_vehicle=type_of_vehicle,
        multiple_deliveries=multiple_deliveries,
        Festival=festival,
        City=city,
        distance=distance
    )

            predict_button = st.button("Predict Delivery Time")
            pred_pipe=PredictionPipeline()
            if predict_button:
                input_data = custom_data.get_as_DataFrame()
                pred=pred_pipe.predict(features=input_data)
                return (f'estimated time of delivery is {pred} in minutes')


            







