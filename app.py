import os,sys
import pandas as pd
import streamlit as st
from src.constants import * # importing all variables from constant file
from src.config.configuration import * # importing all from 
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

feature_engineerin_path=FEATURE_ENGINEERING_FILE_PATH
transformer_file_path=PREPROSECCING_OBJ_FILE
model_file_path=MODEL_FILE_PATH


st.write("# Welcome to Zomoto Delivery Prediction app 👋")



st.markdown(
    """
    This streamlit app offers  3  function like shown in sidebar.
    **👈 Select a demo from the sidebar** to see some examples
    of what This app  can do!
    ### Want to learn more?
    - Check out [GitHub](https://github.com/kalehariprasad/ML-modular-coding-project)
    
"""
)
nav_selection = st.sidebar.radio("Navigation", ["train", "Batch Prediction","Single Prediction"])
if nav_selection=='train':
    Streamlit.train()
    st.write("training compleeted")
elif nav_selection == "Single Prediction":   
    Streamlit.single_prediction()

    


