import argparse
import datetime
import glob

import joblib
import numpy as np
import cianparser
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import logging

from sklearn.model_selection import train_test_split

logger = logging.getLogger('lifecycle')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('lifecycle.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

TRAIN_SIZE = 0.2
MODEL_NAME = "./models/model.pkl"

raw_data_path = './data/raw'
processed_data_path = './data/processed'
X_cols = ['total_meters', 'floor', 'floors_count', 'rooms_count']
y_cols = ['price']
