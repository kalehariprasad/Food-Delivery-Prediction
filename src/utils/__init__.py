from src.logger import logging
from src.exception import CustomeExeption
import os, sys
import pickle
from sklearn.metrics import r2_score

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj )
    except Exception as e:
        raise CustomeExeption(e, sys)
    
def model_evaluation(X_train,y_train,X_test,y_test,models):
    try:

        report={}
        for model_name, model_instance in models.items():
            model_instance.fit(X_train, y_train)
            y_test_pred = model_instance.predict(X_test)
            model_r2_score = r2_score(y_test, y_test_pred)
            report[model_name] = model_r2_score
        return report


    except Exception as e:
        raise CustomeExeption(e, sys)