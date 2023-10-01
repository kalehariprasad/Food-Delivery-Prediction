import os,sys
from src.constants import * 
from src.config.configuration import *
from src.exception import CustomeExeption
from src.logger import logging
from dataclasses import dataclass
from src.utils import load_model
from src.pipeline.training_pipeline import Train
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline
from prediction.batch import BatchPrediction
import pandas as pd
import streamlit as st

class Streamlit:
    def __init__(self):
       pass

    def train(self):
        st.title('Zomato Delivery Prediction training stage')
        train=Train()
        train.main()
     

    def batchprediction(self):
        st.title("batch  Delivery prediction ")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            batch=BatchPrediction()
            batch.start_batch_prediction(input_file=df)
            
            
    


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
                st.text(f'Estimated time of delivery is {pred} in minutes')


            







