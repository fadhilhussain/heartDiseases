import os
from src.exception import CustomException
from src.logger import logging 
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    