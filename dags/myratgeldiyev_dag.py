import sys
import os
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
import logging


SPLIT_SIZE = 0.2
MODEL_NAME = "model.pkl"

BASE_DIR = os.path.dirname(__file__)
MY_DIR = os.path.join(BASE_DIR, "myratgeldiyev")
sys.path.append(BASE_DIR)
REQUIREMENTS_FILE = "myratgeldiyev/requirements.txt"




os.environ["UV_CACHE_DIR"] = "/opt/airflow/dags/myratgeldiyev/.uv_cache"


def get_db_params():
    hook = PostgresHook(postgres_conn_id='tutorial_pg_conn')
    conn = hook.get_conn()
    params = {
        'host': conn.info.host,
        'port': conn.info.port,
        'dbname': conn.info.dbname,
        'user': conn.info.user,
        'password': conn.info.password
    }
    conn.close()
    return params


default_args = {
    "owner": "ashyr"
    # "retries": 0
    # "retry_delay": timedelta(minutes=2),
}

@dag(
    dag_id="myratgeldiyev-dag",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 6, 1),
    catchup=False,
    tags=["house-price-predictor"],
)
def project_dag():
    pg_params = get_db_params()
    task_params = {
        'my_dir': MY_DIR,
        'db_params': pg_params,
        'split_size': SPLIT_SIZE,
        'model_name': MODEL_NAME,
        'X_cols': ['total_meters', 'floor', 'floors_count', 'rooms_count'],
        'y_cols': ['price']

    }

    @task(task_id="create-table")
    def create_table_if_not_exists():
        hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
        table_name = "apartments_myratgeldiyev"

        # Проверяем наличие таблицы
        result = hook.get_first(
            f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
            """, parameters=(table_name,)
        )

        if result and not result[0]:
            hook.run(f"""
            CREATE TABLE {table_name} (
                url_id TEXT PRIMARY KEY,
                total_meters NUMERIC,
                price NUMERIC,
                floor INTEGER,
                floors_count INTEGER,
                rooms_count INTEGER
            );
            """)

    @task.virtualenv(
        requirements=REQUIREMENTS_FILE,
        system_site_packages=False,
    )
    def run_parse(params):
        import datetime
        import logging
        import cianparser
        import pandas as pd


        my_dir = params['my_dir']

        logger = logging.getLogger('lifecycle')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(my_dir, 'lifecycle.log'), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("parsing cian apartments")
        moscow_parser = cianparser.CianParser(location="Москва")
        t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        n_rooms = (1, 2, 3)
        csv_path = f'{my_dir}/{t}_{'_'.join(map(str, n_rooms))}.csv'
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=n_rooms,
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 1,
                "object_type": "secondary"
            })
        df = pd.DataFrame(data)

        df.to_csv(csv_path,
                  encoding='utf-8',
                  index=False)

    # noinspection PyUnresolvedReferences
    @task.virtualenv(
        requirements=REQUIREMENTS_FILE,
        system_site_packages=False,
    )
    def run_preprocess(params):
        import glob
        import pandas as pd
        from sqlalchemy import create_engine, Table, MetaData
        from sqlalchemy.dialects.postgresql import insert
        import logging

        my_dir = params['my_dir']
        db_params = params['db_params']

        logger = logging.getLogger('lifecycle')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(my_dir, 'lifecycle.log'), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info('preprocessing data')


        file_list = glob.glob(my_dir + "/*.csv")
        logger.info(f"found files: {file_list}")
        main_dataframe = pd.read_csv(file_list[0], delimiter=',')
        for i in range(1, len(file_list)):
            data = pd.read_csv(file_list[i], delimiter=',')
            df = pd.DataFrame(data)
            main_dataframe = pd.concat([main_dataframe, df], axis=0)

        main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
        required_cols = ['url_id', 'total_meters', 'price', 'floor', 'floors_count', 'rooms_count']
        main_dataframe.dropna(subset=required_cols, inplace=True)
        main_dataframe.drop_duplicates(subset=['url_id'], keep='last', inplace=True)
        data = main_dataframe[['url_id', 'total_meters', 'price', 'floor', 'floors_count', 'rooms_count']]
        data = data[data['price'] < 100_000_000]
        data.sort_values('url_id', inplace=True)

        user = db_params['user']
        password = db_params['password']
        host = db_params['host']
        port = db_params['port']
        db_name = db_params['dbname']

        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}")
        metadata = MetaData()
        apartments_table = Table('apartments_myratgeldiyev', metadata, autoload_with=engine)

        batch_size = 100  # размер батча, можно увеличить

        with engine.begin() as conn:
            for start in range(0, len(data), batch_size):
                batch_df = data.iloc[start:start + batch_size]
                records = batch_df.to_dict(orient='records')

                stmt = insert(apartments_table).values(records)
                do_update_stmt = stmt.on_conflict_do_update(
                    index_elements=['url_id'],
                    set_={
                        'total_meters': stmt.excluded.total_meters,
                        'price': stmt.excluded.price,
                        'floor': stmt.excluded.floor,
                        'floors_count': stmt.excluded.floors_count,
                        'rooms_count': stmt.excluded.rooms_count,
                    }
                )
                conn.execute(do_update_stmt)

        for file_path in file_list:
            try:
                os.remove(file_path)
                logger.info(f"deleted: {file_path}")
            except Exception as e:
                logger.info(f"failed to delete {file_path}: {e}")

    # noinspection PyUnresolvedReferences
    @task.virtualenv(
        requirements=REQUIREMENTS_FILE,
        system_site_packages=False,
    )
    def run_train(params):
        import joblib
        import logging
        import pandas as pd
        import numpy as np
        from catboost import CatBoostRegressor, Pool
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sqlalchemy import create_engine

        my_dir = params['my_dir']
        split_size = params['split_size']
        model_name = params['model_name']
        db_params = params['db_params']
        X_cols = params['X_cols']
        y_cols = params['y_cols']

        logger = logging.getLogger('lifecycle')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(my_dir, 'lifecycle.log'), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("Connecting to PostgreSQL and loading data")
        user = db_params['user']
        password = db_params['password']
        host = db_params['host']
        port = db_params['port']
        db_name = db_params['dbname']

        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}")

        df = pd.read_sql("SELECT * FROM apartments_myratgeldiyev", engine)
        df.dropna(subset=X_cols + y_cols, inplace=True)

        logger.info(f"Loaded {len(df)} records from the database.")

        X = df[X_cols]
        y = df[y_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split_size,
            shuffle=False
        )

        train_pool = Pool(X_train, label=y_train)
        test_pool = Pool(X_test, label=y_test)

        model_params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'verbose': False
        }

        model = CatBoostRegressor(**model_params)
        logger.info("Training CatBoost model")
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Model RMSE: {rmse:.2f}")
        logger.info(f"Model R²: {r2:.4f}")

        model_path = f"{my_dir}/{model_name}"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")


    create_table_if_not_exists() >> run_parse(task_params) >> run_preprocess(task_params) >> run_train(task_params)

project_dag = project_dag()