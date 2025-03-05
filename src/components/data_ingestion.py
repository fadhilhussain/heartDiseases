import os
import sys
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging 
from dataclasses import dataclass
from src.components.feature_selection import FeatureSelection
from src.components.data_transformation import DataTransormation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts','data.csv')
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info('initializing ingestion part')
        try:
            df = pd.read_csv('notebook/data/data_cleaned1.csv')
            logging.info('read dataFrame')

            #make 'artifacts' folder
            os.makedirs(os.path.dirname(self.data_config.raw_data_path),exist_ok=True)

            #save dataset to artifacts file
            df.to_csv(self.data_config.raw_data_path,index=False,header=True)

            logging.info('data ingestion completed')
            
            return (
                self.data_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
    def data_splitting(self,data):
        try:
            df = pd.read_csv(data)

            train_data,test_data = train_test_split(df,test_size=0.3,random_state=42,stratify=df['TenYearCHD'])

            #save to artifacts folder 
            train_data.to_csv(self.data_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_config.test_data_path,index=False,header=True)

            logging.info('train and test split done')
            return(
                self.data_config.train_data_path,
                self.data_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

        

if __name__ == '__main__':
    object = DataIngestion()
    raw_data = object.initiate_ingestion()
    
    #feature selection 
    feature_selection = FeatureSelection()
    selected_feature_data = feature_selection.feature_selection(raw_data,threshold=0.01)
    train_data, test_data = object.data_splitting(selected_feature_data)

    #data transformation
    data_transformation = DataTransormation()
    train_array, test_array = data_transformation.initiate_transformation(train_data,test_data)

    #model training 
    model_trainer = ModelTrainer()
    print(model_trainer.model_trainer(train_array,test_array))
    