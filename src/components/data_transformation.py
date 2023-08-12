import os,sys
from src.constants import * # importing all variables from constant file
from src.config.configuration import * # importing all from 
from src.logger import logging
from src.exception import CustomeExeption
from src.utils import save_obj
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class featureengineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        logging.infO("feature engineering started")

    def diastance_numpy(seld,df,lat1,long1,lat2,long2):
        p=np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['distance'] = 12734 * np.arccos(np.sort(a))

        return df
    

    def Transform_data(self,df):
        try:
            df=pd.read_csv(DATASET_PATH)
            df.drop("ID",axis=1)
            self.diastance_numpy*(df,'Restaurant_latitude',
                                  'Restaurant_longitude',
                                  'Delivery_location_latitude',
                                  'Delivery_location_longitude')
            df.dro(['Delivery_person_ID','Restaurant_latitude',
                                  'Restaurant_longitude',
                                  'Delivery_location_latitude',
                                  'Delivery_location_longitude','Order_Date','Time_Orderd','Time_Order_picked'])
            logging.info('dropped columns from data frame and created new column called distance')
        except Exception as e:
            raise CustomeExeption(e,sys)
        
        def transform(sef,x:pd.DataFrame,y:None):
            try:
                transformed_df=self.Transform_data(x)
                return transformed_df
            except Exception as e:
                raise CustomeExeption(e,sys)


@dataclass            
class DataTransformationconfig():
    processed_obj_file_path=PREPROSECCING_OBJ_FILE
    featureengineering_obj_path=FEATURE_ENGINEERING_FILE_PATH
    transformed_train_file=TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_file=TRANSFORMED_TEST_FILE_PATH

class Datatransformation():
    def __init__(self) :
        self.data_transformation_config=DataTransformationconfig()

