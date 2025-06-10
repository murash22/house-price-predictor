import pandas as pd
from dotenv import dotenv_values
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth, HTTPTokenAuth
from logging.config import dictConfig
from dataclasses import dataclass
import joblib
# import configs

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
cfg = dotenv_values('.env')
auth = HTTPTokenAuth(scheme='Bearer')
tokens = {cfg['APP_TOKEN']: "user1"}
CORS(app, resources={r"/api/*": {"origins": "*"}})

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

@dataclass
class HouseInfo:
    area: int = None
    rooms: int = None
    total_floors: int = None
    floor: int = None

# Сохранение модели
from src.download_from_s3 import download_file
folder = "models"
model_name = "catboost_v1.pkl"
download_file(folder, model_name)
model_path = f'{folder}/{model_name}'

loaded_model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/numbers', methods=['POST'])
@auth.login_required
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
        predicted = loaded_model.predict(input_df)
        price = predicted[0]
        price = int(price)
    except ValueError:
        return {'status': 'error', 'data': 'internal server error'}
    return {'status': 'success', 'data': price}


if __name__ == '__main__':
    app.run(debug=True)