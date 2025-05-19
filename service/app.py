import pandas as pd
from flask import Flask, render_template, request
from logging.config import dictConfig
from dataclasses import dataclass
import joblib
import xgboost as xgb

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

@dataclass
class HouseInfo:
    area: int = None
    rooms: int = None
    total_floors: int = None
    floor: int = None

model_path = 'models/xgboost_v1.pkl'

loaded_model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    data = request.get_json()
    app.logger.info(f'Requst data: {data}')
    try:
        input_data = {
            'total_meters': [float(data['area'])],
            'floor': [int(data['floor'])],
            'floors_count': [int(data['total_floors'])],
            'rooms_count': [int(data['rooms'])],
        }
        input_df = pd.DataFrame(input_data)
        dmatrix_input = xgb.DMatrix(input_df)
        predicted = loaded_model.predict(dmatrix_input)
        price = predicted[0]
        price = int(price)
    except ValueError as e:
        app.logger.info(e)
        return {'status': 'error', 'data': 'internal server error'}
    return {'status': 'success', 'data': price}


if __name__ == '__main__':
    app.run(debug=True)