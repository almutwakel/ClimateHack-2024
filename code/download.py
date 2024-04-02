import os
from datetime import datetime, time, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
from ocf_blosc2 import Blosc2
from torch.utils.data import DataLoader, IterableDataset
import dask
import matplotlib.pyplot as plt
import tqdm
import pathlib
import numpy as np
import pandas as pd
import gc
import zarr

def get_day_slice(date, data):
    data_slice = data.loc[
        {
            # 7am to 5pm
            "time": slice(
                date + timedelta(hours=7),
                date + timedelta(hours=17),
            )
        }
    ]
    # sometimes there is no data
    if len(data_slice.time) == 0:
        return None
    return data_slice

data_months = [  # (year, month)
                 ("2020", "1"),
                 ("2020", "2"),
                 ("2020", "3"),
                 ("2020", "4"),
                 ("2020", "5"),
                 ("2020", "6"),
                 ("2020", "7"),
                 ("2020", "8"),
                 ("2020", "9"),
                 ("2020", "10"),
                 ("2020", "11"),
                 ("2020", "12"),
                 ("2021", "1"),
                 ("2021", "2"),
                 ("2021", "3"),
                 ("2021", "4"),
                 ("2021", "5"),
                 ("2021", "6"),
                 ("2021", "7"),
                 ("2021", "8"),
                 ("2021", "9"),
                 ("2021", "10"),
                 ("2021", "11"),
                 ("2021", "12"),


]

if __name__ == "__main__":
    
    if not os.path.exists("submission"):
        os.makedirs("submission", exist_ok=True)

        os.system("curl -L https://raw.githubusercontent.com/climatehackai/getting-started-2023/main/submission/competition.py --output submission/competition.py")
        os.system("curl -L https://raw.githubusercontent.com/climatehackai/getting-started-2023/main/submission/doxa.yaml --output submission/doxa.yaml")
        os.system("curl -L https://raw.githubusercontent.com/climatehackai/getting-started-2023/main/submission/model.py --output submission/model.py")
        os.system("curl -L https://raw.githubusercontent.com/climatehackai/getting-started-2023/main/submission/run.py --output submission/run.py")
        os.system("curl -L https://raw.githubusercontent.com/climatehackai/getting-started-2023/main/indices.json --output indices.json")

    if not os.path.exists("data"):
        os.makedirs("data/pv/2020", exist_ok=True)
        os.makedirs("data/pv/2021", exist_ok=True)
        os.makedirs("data/satellite-hrv/2020", exist_ok=True)
        os.makedirs("data/satellite-hrv/2021", exist_ok=True)
        os.makedirs("data/weather/2020", exist_ok=True)
        os.makedirs("data/weather/2021", exist_ok=True)
        os.system("curl -L https://huggingface.co/datasets/climatehackai/climatehackai-2023/resolve/main/pv/metadata.csv --output data/pv/metadata.csv")
        for (year, month) in data_months:
            os.system('curl -L "https://huggingface.co/datasets/climatehackai/climatehackai-2023/resolve/main/pv/"$year"/"$month".parquet" --output "data/pv/"$year"/"$month".parquet"')
            os.system('curl -L "https://huggingface.co/datasets/climatehackai/climatehackai-2023/resolve/main/satellite-hrv/"$year"/"$month".zarr.zip" --output "data/satellite-hrv/"$year"/"$month".zarr.zip"')
            os.system('curl -L "https://huggingface.co/datasets/climatehackai/climatehackai-2023/resolve/main/weather/"$year"/"$month".zarr.zip" --output "data/weather/"$year"/"$month".zarr.zip"')
        os.system("cp indices.json data")

    for year in [2020, 2021]:
        for month in range(1, 13):
            if not os.path.exists("data/satellite-hrv/{year}/{month}"):
                hrv_data_month = xr.open_dataset(
                    f"data/satellite-hrv/{year}/{month}.zarr.zip",
                    engine="zarr", chunks="auto"
                )
                start_date = datetime(year, month, 1)
                end_date = start_date + relativedelta(months=1)

                cur = start_date
                days_to_get = []
                while cur != end_date + timedelta(days=1):
                    days_to_get.append(cur)
                    cur = cur + timedelta(days=1)

                for date in tqdm.tqdm(days_to_get):
                    slc = get_day_slice(date, hrv_data_month)
                    if slc is None:
                        continue
                    combined = xr.concat([slc], dim='time')
                    times = combined['time'].to_numpy()
                    data = combined['data'].to_numpy()
                    day = date.day
                    pathlib.Path(f"data/satellite-hrv/{year}/{month}").mkdir(parents=False, exist_ok=True)
                    p = pathlib.Path(f'data/satellite-hrv/{year}/{month}/{day}.npz')
                    if p.exists():
                        raise ValueError(f'Path {p} already exists!')
                    print('Saving', p)
                    np.savez(
                        p,
                        times=times,
                        data=data,
                    )
                    del data
                    del times
                    gc.collect()
                del hrv_data_month

