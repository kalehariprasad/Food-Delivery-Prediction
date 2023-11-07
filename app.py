import os,sys
import pandas as pd
import numpy as np
import streamlit as st
from src.constants import * 
from src.config.configuration import * 
from src.logger import logging
from src.exception import CustomeExeption
from src.components.data_transformation import DataTransformation,featureengineering
from src.components.model_trainer import ModelTrainer
from src.components.data_ingection import DataIngection
from src.pipeline.training_pipeline import Train
from prediction.batch import BatchPrediction
from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData
from src.utils import load_model
class Streamlit:
    def __init__(self):
       pass

    def train(self):
        st.title('Food Delivery Prediction training stage')
        train=Train()
        train.main()
     

    def batchprediction(self):
        st.title("batch  Delivery prediction ")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            batch=BatchPrediction()
            pred=batch.initiate_batch_prediction(input_file=df)
            st.write('prediction are')
            st.write(pd.DataFrame(pred)) 
            
            
    


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
            input_df= custom_data.get_as_DataFrame()

  
            predict_button = st.button("Predict Delivery Time")
            if predict_button:
                    pred_fun=PredictionPipeline()
                    pre = pred_fun.predict(input_df)
                    st.text(f'Estima ted time of delivery is in minutes : {pre[0]} minutes')


nav_selection = st.sidebar.radio("Navigation", ["Main Page", "Train", "Batch Prediction", "Single Prediction"])
streamlit_instance = Streamlit()
if nav_selection == 'Train':
    streamlit_instance.train()
    st.write("Training completed")
elif nav_selection == "Batch Prediction":
    streamlit_instance.batchprediction()
elif nav_selection == "Single Prediction":
    streamlit_instance.single_prediction()
elif nav_selection == "Batch Prediction":
    # Add logic for Batch Prediction page
    pass
else:
    # Main Page content
    st.title("Welcome to Food  Delivery Prediction app 👋")
    st.write("This streamlit app offers 3 functions like shown in the sidebar.")
    st.write("**👈 Select a demo from the sidebar** to see some examples of what This app can do!")
    st.write("## Want to learn more?")
    st.write("- Check out [GitHub](https://github.com/kalehariprasad/ML-modular-coding-project)")

    


