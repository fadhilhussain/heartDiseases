import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from dataclasses import dataclass
from src.utils import split_features,save_object
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessed_data_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransormation:
    def __init__(self):
        self.preprocess_data_config = DataTransformationConfig()

    def data_smote(self,data):
        try:
            X_train, y_train = split_features(
                data=data
            )
            logging.info('applying smote')
            smote = SMOTE()
            X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
            return(
                X_train_smote,
                y_train_smote
            )
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_tranformer_object(self):
        try:
            columnss = [
                'gender',
                'age',
                'education',
                'currentSmoker',
                'cigsPerDay',
                'prevalentHyp',
                'totChol',
                'sysBP',
                'BMI',
                'heartRate',
                'glucose'
            ]
            feature_pipeline = Pipeline(
                steps=[
                    ('scaler',StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer(
                [

                ('pipeline',feature_pipeline,columnss)
                
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train,y_train = self.data_smote(data=train_df)
            X_test, y_test = split_features(
                data=test_df
            )
            logging.info('appplying preprocessor object')
            processor_object = self.get_tranformer_object()

            input_train_data = processor_object.fit_transform(X_train)
            input_test_data = processor_object.transform(X_test)

            train_arr = np.c_[
                input_train_data, np.array(y_train)
            ]
            test_arr = np.c_[
                input_test_data, np.array(y_test)
            ]

            logging.info('save processing object')
            save_object(
                file_path=self.preprocess_data_config.preprocessed_data_path,
                obj=processor_object
            )
            
            return(
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)






