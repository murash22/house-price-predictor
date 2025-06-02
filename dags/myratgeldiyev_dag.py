from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "pabd25",
    "depends_on_past": False,
    "start_date": datetime(2025, 6, 2),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="myratgeldiyev-dag",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    tags=["house-price-predictor", "model"]
) as dag:

    parse = BashOperator(
        task_id="myratgeldiyev-dag-parse",
        bash_command="/home/pabd25/myratgeldiyev/house-price-predictor/venv/bin/python3 /home/pabd25/myratgeldiyev/house-price-predictor/src/parse_data.py"
    )

    preprocess_data = BashOperator(
        task_id="myratgeldiyev-dag-preprocess",
        bash_command="/home/pabd25/myratgeldiyev/house-price-predictor/venv/bin/python3 /home/pabd25/myratgeldiyev/house-price-predictor/src/preprocess.py"
    )

    train_model = BashOperator(
        task_id="myratgeldiyev-dag-train",
        bash_command="/home/pabd25/myratgeldiyev/house-price-predictor/venv/bin/python3 /home/pabd25/myratgeldiyev/house-price-predictor/src/train.py"
    )

    # test_model = BashOperator(
    #     task_id="myratgeldiyev-dag-test",
    #     bash_command="python ~/martirosyan/pabd/src/train.py"
    # )
    parse >> preprocess_data >> train_model