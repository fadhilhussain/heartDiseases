import os 
import sys
import dill
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


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
    
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        for i in range(len(list(models))):

            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params[model_name]

            random = RandomizedSearchCV(model,param,cv=cv,n_jobs=-1)
            random.fit(X_train,y_train)
            print(f'Best params of  {model_name} : {random.best_estimator_}')
            logging.info('fit the model')

            #assigning new params to model
            model.set_params(**random.best_params_)
            model.fit(X_train,y_train) 

            #predict both train and test with threshold
            y_pred_train = model.predict(X_train)

            #setting threshold 
            y_probs = model.predict_proba(X_test)[:,-1]

            precision_,recall_,threshold = precision_recall_curve(y_test,y_probs)
            f1_scores = 2 * (precision_ * recall_)/(precision_ + recall_ + 1e-10)

            #gets best threshold 
            best_threshold = threshold[np.argmax(f1_scores)]

            #pred value by custom thresold 
            y_pred_custom = (y_probs >= best_threshold).astype(int)

            #evaluations 
            train_model_accuray = roc_auc_score(y_train,y_pred_train)
            test_model_accuracy = roc_auc_score(y_test,y_pred_custom)

            report[model_name] = test_model_accuracy

            return report
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)