from src.logger import logging
from src.exception import CustomeExeption
import os, sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def get_unique(df):
    cat_cols=df.select_dtypes("object")
    for col in cat_cols:
        unique_values = df[col].unique()
    print(f"Unique values in column '{col}': {unique_values}")

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj )
    except Exception as e:
        raise CustomeExeption(e, sys)
    
def model_evaluation(X_train,y_train,X_test,y_test,models,):
    
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
           # para=param[list(models.keys())[i]]

            #gs = GridSearchCV(model,para,cv=3)
            #gs.fit(X_train, y_train)

            #model.set_params(**gs.best_params_)
            #model.fit(X_train, y_train)

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
            logging.info("exception has occured while saving object")
            raise CustomeExeption(e,sys)

def load_model(file_path):
    try:
        with open(file_path,"rb") as f:
             return pickle.load(f)
    except Exception as e:
            logging.info('exception has ocuured during loading object')
            raise CustomeExeption(e,sys)


