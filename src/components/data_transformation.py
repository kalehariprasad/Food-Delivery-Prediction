import os,sys
from src.constants import * # importing all variables from constant file
from src.config.configuration import * # importing all from 
from src.logger import logging
from src.exception import CustomeExeption
from src.utils import save_obj,get_unique
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class featureengineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logging.info("feature engineering started")

    def diastance_numpy(self, df, lat1, long1, lat2, long2):
        p = np.pi / 180
        a = 0.5 - np.cos((df[lat2] - df[lat1]) * p) / 2 + np.cos(df[lat1] * p) * np.cos(df[lat2] * p) * (1 - np.cos((df[long2] - df[long1]) * p)) / 2
        df['distance'] = 12734 * np.arccos(np.sort(a))
        
        return df
    
    def transform(self, df):
        try:
            df = pd.read_csv(DATASET_PATH)
            df=df.drop(["ID"], axis=1)
            self.diastance_numpy(df, 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude')
            df=df.drop(['Delivery_person_ID', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Order_Date', 'Time_Orderd', 'Time_Order_picked'], axis=1)
            logging.info('dropped columns from data frame and created a new column called distance')
            
          
            return df   
        except Exception as e:
            raise CustomeExeption(e,sys)
                
                
    def fit(self,x,y=None):
        return self
    

    def transform_data(self,x:pd.DataFrame,y:None):
        try:
            transformed_df=self.transform_data(x)
            return transformed_df
        except Exception as e:
            raise CustomeExeption(e,sys)
        
@dataclass
class DataTransformationConfig():
    processed_obj_file_path=PREPROSECCING_OBJ_FILE
    featureengineering_obj_path=FEATURE_ENGINEERING_FILE_PATH
    fe_train_path=FE_TRAIN_PATH
    fe_test_path=FE_TEST_PATH
    transformed_train_file=TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_file=TRANSFORMED_TEST_FILE_PATH

class DataTransformation:
    def __init__(self):
        self.data_transformatio_config=DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            road_traffic_density=['Low','Medium', 'High','Jam']
            weather_condition=['sunny','Cloudy','Fog','Sandstorms','Windy','Stormy']
            categorical_columns = ['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encoder = ['Road_traffic_density', 'Weather_conditions']
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance']
            # Numerical pipeline
            numerical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Categorical Pipeline
            categorical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
             # ordinal Pipeline
            ordinal_pipeline = Pipeline(steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('ordinal', OrdinalEncoder(categories=[road_traffic_density, weather_condition], handle_unknown='use_encoded_value', unknown_value=-1)),
                    ('scaler', StandardScaler(with_mean=False))
            ])

            
            preprocssor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline,numerical_column ),
                ('categorical_pipeline', categorical_pipeline,categorical_columns ),
                ('ordinal_pipeline', ordinal_pipeline,ordinal_encoder )
            ])

            logging.info("Pipeline Steps Completed")
            return preprocssor
                
        except Exception as e:
            raise CustomeExeption(e,sys)

    def getfeature_engineering_object(self):
        try:
            feature_engineering_object=Pipeline(
                steps=[("feature engineering",featureengineering()),
                ]
            )
            return feature_engineering_object
        except Exception as e:
            raise CustomeExeption(e,sys)
    def intiate_data_transformation(self,train_df,test_df):
        try:
            
            logging.info('obtaining feature enfineering object')
            fe_obj=self.getfeature_engineering_object()
            train_feature_engineering=fe_obj.fit_transform(train_df)
            os.makedirs(os.path.dirname(self.data_transformatio_config.fe_train_path),exist_ok=True)
            train_feature_engineering.to_csv(self.data_transformatio_config.fe_train_path,index=True)

            test_featuere_engineering=fe_obj.transform(test_df)
            os.makedirs(os.path.dirname(self.data_transformatio_config.fe_test_path),exist_ok=True)
            test_featuere_engineering.to_csv(self.data_transformatio_config.fe_test_path,index=True)


            preprocessing_object=self.get_data_transformation_object()


            target_column="Time_taken (min)"
            X_train=train_feature_engineering.drop(columns=target_column,axis=1)
            y_train=train_feature_engineering[target_column]
            X_test=test_featuere_engineering.drop(columns=target_column,axis=1)
            y_test=test_featuere_engineering[target_column]

            X_train=preprocessing_object.fit_transform(train_feature_engineering)
            X_test=preprocessing_object.transform(test_featuere_engineering)

            train_arry = np.c_[X_train, y_train]  
            test_arry = np.c_[X_test, y_test]
            
            df_train=pd.DataFrame(train_arry)
            df_test=pd.DataFrame(test_arry)
            os.makedirs(os.path.dirname(self.data_transformatio_config.transformed_train_file),exist_ok=True)
            df_train.to_csv(self.data_transformatio_config.transformed_train_file,index=False,header=True)
            os.makedirs(os.path.dirname(self.data_transformatio_config.transformed_test_file),exist_ok=True)
            df_test.to_csv(self.data_transformatio_config.transformed_test_file,index=False,header=True)
            save_obj(file_path=self.data_transformatio_config.featureengineering_obj_path,obj=fe_obj)
            save_obj(file_path=self.data_transformatio_config.processed_obj_file_path,obj=preprocessing_object)
           
            return (train_arry,test_arry)





        except Exception as e:
            raise CustomeExeption(e,sys)
