import os 
import sys
import dill
import pandas as pd
from src.logger import logging
from src.exception import CustomException

def split_features(data):
    try:
        target = 'TenYearCHD'
        X = data.drop(columns=[target],axis=1)
        y = data[target]

        return X,y
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)