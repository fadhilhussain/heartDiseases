import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.utils import evaluate_model,save_object


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_path_config = ModelTrainerConfig()

    def model_trainer(self,train_array,test_array):
        try:
            logging.info('splitting data in model trainer')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Logistic Regression' : LogisticRegression(),
                'K-nearest neighbor' : KNeighborsClassifier(),
                'XGBoost Classifier' : XGBClassifier(),
                'CatBoost Classifier' : CatBoostClassifier()
            }
            logging.info('defining params variable')
            params = {
                'Logistic Regression' : {
                    'penalty' : ['l1','l2'],
                    'C' : [0.0001,0.001,0.01],
                    'solver' : ['saga'],
                    'max_iter' : [100,200,300],
                    'class_weight' : ['balanced',{0:1,1:2},{0:1,1:3},{0:1,1:5}]
                },
                'K-nearest neighbor' : {
                    'n_neighbors' : list(range(1,50,2)),
                    'weights' : ['uniform','distance'],
                    'metric' : ['euclidean','manhattan','minkowski'],
                    'algorithm' : ['ball_tree','kd_tree','brute']
                },
                'XGBoost Classifier' : {
                    'n_estimators' : [100,200,300,500,600],
                    'learning_rate' : [0.01,0.05,0.1,0.2],
                    'max_depth' : [3,4,5,6,7],
                    'subsample' : [0.6,0.7,0.8,1.0],
                    'reg_lambda' : [0,1,2,3,5],
                    'min_child_weight' : [1,3,5,7],
                    'colsample_bytree' : [0.6,0.7,0.8,1.0],
                    'gamma' : [0,0.1,0.3,0.5],
                    'reg_alpha' : [0,0.1,0.5,1]
                },
                'CatBoost Classifier' : {
                    'depth' : [4,6,8,10],
                    'learning_rate' : [0.01,0.03,0.1,0.2],
                    'iterations' : [100,200,500,1000],
                    'l2_leaf_reg' : [1,3,5,7],
                    'border_count' : [32,64,128],
                    'bagging_temperature' : [0,0.5,1,2],
                    'scale_pos_weight' : [1,1.5,2],
                    'random_strength' : [1,5,10]
                }

            }

            model_report : dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            #best model recall score 
            best_model_score = max(sorted(model_report.values()))

            #best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            #pick the best model 
            best_model = models[best_model_name]

            #setting some threshold
            if best_model_score < 0.3:
                raise CustomException('No best model found')
            
            logging.info(f'best model is {best_model_name}')

            save_object(
                file_path=self.model_path_config.model_path,
                obj=best_model
            )
            return best_model_score
        
        except Exception as e:
            raise CustomException(e,sys)