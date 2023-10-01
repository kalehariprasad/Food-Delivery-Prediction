import os,sys
import pandas as pd
import streamlit as st
from src.constants import * 
from src.config.configuration import * 
from src.logger import logging
from src.exception import CustomeExeption
from src.components.data_transformation import DataTransformation,featureengineering
from src.components.model_trainer import ModelTrainer
from src.components.data_ingection import DataIngection
from src.pipeline.training_pipeline import Train
from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData,Streamlit

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

feature_engineerin_path = FEATURE_ENGINEERING_FILE_PATH
transformer_file_path = PREPROSECCING_OBJ_FILE
model_file_path = MODEL_FILE_PATH



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
    st.write("Welcome to Zomoto Delivery Prediction app 👋")
    st.write("This streamlit app offers 3 functions like shown in the sidebar.")
    st.write("**👈 Select a demo from the sidebar** to see some examples of what This app can do!")
    st.write("### Want to learn more?")
    st.write("- Check out [GitHub](https://github.com/kalehariprasad/ML-modular-coding-project)")

    


