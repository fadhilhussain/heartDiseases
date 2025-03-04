import os
import sys
import pandas as pd
from src.logger import logging 
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import split_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass
class Feature_selection_Config():
    features_data_path: str = os.path.join('artifacts','features.csv')

class FeatureSelection():
    def __init__(self):
        self.feature_path = Feature_selection_Config()
    
    def feature_selection(self,data,threshold):
        try:

            #independent and dependent feature
            X, y = split_features(
                data = data
            )

            ## feature selection by TreeBased Feature Importance
            modelRF = RandomForestClassifier()
            modelRF.fit(X,y)
            feature_importance = modelRF.feature_importances_
            selected_features = X.columns[feature_importance>threshold].to_list()
            logging.info('TreeBased Feature Importance is done ')

            ## feature selection by variance inflation factor
            blood_pressure = ['sysBP','diaBP'] #proven

            Vfi_features  = X[blood_pressure]
            vfi_dataFrame = pd.DataFrame({'Bp':Vfi_features.columns})
            vfi_dataFrame['score'] = [variance_inflation_factor(Vfi_features.values,i)for i in range(Vfi_features.shape[1])]
            
            #check feature impotance 
            if all(vfi_dataFrame['score']==vfi_dataFrame['score'][0]):
                fscore,_ = f_classif(Vfi_features,y)
                removing_feature = blood_pressure[fscore.argmin()]
            logging.info('VFI feature selection is done')

            selected_features.remove(removing_feature)

            new_data = pd.DataFrame(X[selected_features])
            new_data['TenYearCHD'] = y

            new_data.to_csv(self.feature_path.features_data_path,index=False,header=True)

            return self.feature_path.features_data_path
        except Exception as e:
            raise CustomException(e,sys)
