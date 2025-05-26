
import joblib

import pandas as pd
from catboost import CatBoostRegressor, Pool

from common_vars import logger, processed_data_path, X_cols, y_cols, MODEL_NAME

from sklearn.model_selection import train_test_split

def train_model(split_size, model_name):
    """Train model and save with MODEL_NAME"""
    logger.info('training model')
    data = pd.read_csv(f"{processed_data_path}/train_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        data[X_cols],
        data[y_cols],
        test_size=split_size,
        shuffle=False
    )

    train_pool = Pool(
        data=X_train,
        label=y_train,
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
    )

    params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'early_stopping_rounds': 50,
        'verbose': 100
    }

    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True
    )

    joblib.dump(model, model_name)
    test_data = X_test.copy()
    test_data['price'] = y_test
    test_data.to_csv(f"{processed_data_path}/test_data.csv")


if __name__ == "__main__":
    train_model(0.2, MODEL_NAME)