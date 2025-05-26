
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from common_vars import logger, processed_data_path, X_cols, y_cols, MODEL_NAME

def test_model(model_name):
    """Test model with new data"""
    logger.info('testing model')
    model = joblib.load(model_name)
    data = pd.read_csv(f"{processed_data_path}/test_data.csv")
    x_test, y_test = data[X_cols], data[y_cols]

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # Вывод метрик качества
    logger.info(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    logger.info(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
    logger.info(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    logger.info(f"Коэффициент детерминации R²: {r2:.6f}")


if __name__ == "__main__":
    test_model(MODEL_NAME)