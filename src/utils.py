import os 
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

def split_features(data):
    try:
        data_df = pd.read_csv(data)

        target = 'TenYearCHD'
        X = data_df.drop(columns=[target],axis=1)
        y = data_df[target]

        return X,y
    except Exception as e:
        raise CustomException(e,sys)
