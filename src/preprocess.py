
import glob
from common_vars import logger, raw_data_path, processed_data_path
import pandas as pd


def preprocess_data():
    """Filter and remove"""
    logger.info('preprocessing data')

    file_list = glob.glob(raw_data_path + "/*.csv")
    logger.info(f"found files: {file_list}")
    main_dataframe = pd.read_csv(file_list[0], delimiter=',')
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i], delimiter=',')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
    data = main_dataframe[['url_id', 'total_meters', 'price', 'floor', 'floors_count', 'rooms_count']].set_index('url_id')
    data = data[data['price'] < 100_000_000]
    data.sort_values('url_id', inplace=True)
    data.to_csv(f"{processed_data_path}/train_data.csv")


if __name__ == "__main__":
    preprocess_data()