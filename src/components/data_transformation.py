import os
import sys
from src.logger import logging
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE
from src.utils import split_features
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessed_data_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransormation:
    def __init__(self):
        self.preprocess_data_config = DataTransformationConfig()

    def data_smote(self,train_data):
        try:
            X_train, y_train = split_features(
                data=train_data
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



