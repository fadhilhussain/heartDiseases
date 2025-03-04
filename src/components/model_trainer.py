import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


# @dataclass
# class ModelTrainerConfig:
#     model_path : 