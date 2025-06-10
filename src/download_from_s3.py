import argparse

from dotenv import dotenv_values
import boto3

BUCKET_NAME = 'pabd25'
YOUR_SURNAME = 'myratgeldiyev'
LOCAL_FILE_PATH = ['models\\catboost_v1.pkl']

config = dotenv_values(".env")


def download_file(folder, name):
    client = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key']
    )
    object_name = f'{YOUR_SURNAME}/' + "models/" + name
    client.download_file(BUCKET_NAME, object_name, folder + "/" + name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
                        help='Input local data files to upload to S3 storage',
                        default=LOCAL_FILE_PATH)
    args = parser.parse_args()
    download_file(*args.input[0].split('/'))
