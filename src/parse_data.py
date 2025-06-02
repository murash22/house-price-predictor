import datetime

import cianparser
import pandas as pd

from common_vars import logger
from common_vars import raw_data_path

def parse_cian():
    """Parse data to data/raw"""
    logger.info("parsing cian apartments")
    moscow_parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    n_rooms = (1, 2, 3, 'studio')
    csv_path = f'{raw_data_path}/{t}_{'_'.join(map(str, n_rooms))}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=n_rooms,
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 50,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)