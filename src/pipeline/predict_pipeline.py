import sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessing_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'

            #file initilizing 
            preprocessor = load_object(preprocessing_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e,sys)

class NewData:
    def __init__(self,
            gender,
            age,
            education,
            currentSmoker,
            cigsPerDay,
            prevalentHyp,
            totChol,
            sysBP,
            BMI,
            heartRate,
            glucose):
        
        self.gender = gender
        self.age = age
        self.education = education
        self.currentSmoker = currentSmoker
        self.cigsPerDay = cigsPerDay
        self.prevalentHyp = prevalentHyp
        self.totChol = totChol
        self.sysBP = sysBP
        self.BMI = BMI
        self.heartRate = heartRate
        self.glucose = glucose

    def get_dataFrame(self):
        try:
            dataframe_dict = {
                'gender' : [self.gender],
                'age' : [self.age],
                'education' : [self.education],
                'currentSmoker' : [self.currentSmoker],
                'cigsPerDay' : [self.cigsPerDay],
                'prevalentHyp' : [self.prevalentHyp],
                'totChol' : [self.totChol],
                'sysBP' : [self.sysBP],
                'BMI' : [self.BMI],
                'heartRate' : [self.heartRate],
                'glucose' : [self.glucose]
            }
            return pd.DataFrame(dataframe_dict)
        
        except Exception as e:
            raise CustomException(e,sys)