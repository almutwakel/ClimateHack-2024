import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
import json
import lightning.pytorch as pl
import dask

debug = False


class ChDataset(torch.utils.data.Dataset):
    def __init__(self, pv_data, meta_data, site_locs, use_hrv=False):
        self.pv_data = pv_data.groupby([pd.Grouper(level='ss_id')])
        self.ss_ids = list(self.pv_data.groups.keys())
        self.meta_data = meta_data
        self.site_locs = site_locs
        self.use_hrv = use_hrv

    def toggle_hrv(self):
        self.use_hrv = not self.use_hrv
        print("use_hrv set to", self.use_hrv)

    def __len__(self):
        return len(self.ss_ids) * 10

    def __getitem__(self, index):
        # Get the data for the ss_id
        ss_id = self.ss_ids[index % len(self.ss_ids)]
        site_data = self.pv_data.get_group(ss_id)
        lat,long = self.meta_data.loc[ss_id, ['latitude_rounded', 'longitude_rounded']].values
        orientation,tilt,kwp = self.meta_data.loc[ss_id, ['orientation', 'tilt', 'kwp']].values

        # Randomly select a timestamp with non-zero generation
        # TODO: test different thresholds / coinflip idea
        non_zero = site_data[site_data['power'] > 0.2]
        if len(non_zero) == 0:
            non_zero = site_data

        # Only use btw 7am to 5pm
        filter = non_zero.index.get_level_values('timestamp').hour.isin(range(7, 17))
        timestamp = np.random.choice(non_zero.index.get_level_values('timestamp')[filter])

        # Add random noise to timestamp up to 1 hours
        timestamp = timestamp - pd.Timedelta(hours=0, minutes=5*np.random.randint(0, 12))

        if self.use_hrv:
            # Get the hrv data for the timestamp
            year, month, day = timestamp.year, timestamp.month, timestamp.day
            fpath = f'data/satellite-hrv/{year}/{month}/{day}.npz'
            if not os.path.exists(fpath):
                if debug:
                    print("Could not find hrv data for day", timestamp)
                return self.__getitem__(index + 1)

            hrv_data_day = np.load(fpath)
            time_index = np.where(hrv_data_day["times"]==np.datetime64(timestamp))[0]

            if time_index.size == 0:
                if debug:
                    print("Could not find matching timestamp", timestamp)
                return self.__getitem__(index + 1)

            time_index = time_index[0]

            hrv_data = hrv_data_day["data"][time_index:time_index+12]
            x, y = self.site_locs["hrv"][ss_id]
            r = 64
            hrv_data = hrv_data[:, y - r : y + r, x - r : x + r, 0][:12]

            if len(hrv_data) < 12:
                if debug:
                    print("Could not create a large enough window", timestamp)
                return self.__getitem__(index + 1)
            elif (np.isnan(hrv_data).any()):
                print("NAN-HRV WARNING")
                return self.__getitem__(index + 1)
        else:
            hrv_data = None

        # Get the data for 5.5 hours (30 min for buffer)
        data = site_data.loc[timestamp:timestamp + pd.Timedelta(hours=5.5)]

        if len(data) < 60:
            if debug:
                print("Could not create a large enough window", timestamp)
            return self.__getitem__(index + 1)

        # First 1 hour is input, last 4 hours is target
        pv, target = data['power'].iloc[:12].values, data['power'].iloc[12:60].values
        if len(pv) != 12 or len(target) != 48:
            return self.__getitem__(index + 1)

        # Normalize metadata
        lat = (lat - 52.979) / 1.478 # normalize to mean 0 std 1
        long = (long - (-1.409)) / 1.389 # normalize to mean 0 std 1
        start_time = data.index.get_level_values('timestamp')[0]
        day_of_year = start_time.dayofyear / 365
        time_of_day = (start_time.hour + start_time.minute / 60) / 24
        orientation = orientation / 360
        tilt = tilt / 90
        kwp = kwp / 10
        # metadata = np.array([lat, long, day_of_year, time_of_day, orientation, tilt, kwp])

        if self.use_hrv:
            return pv, lat, long, day_of_year, time_of_day, orientation, tilt, kwp, hrv_data, target
        else:
            return pv, lat, long, day_of_year, time_of_day, orientation, tilt, kwp, target

def collate_fn(batch):
    if len(batch[0]) == 10:
        return (
            torch.stack([torch.from_numpy(x[0]) for x in batch]),
            torch.tensor([x[1] for x in batch]),
            torch.tensor([x[2] for x in batch]),
            torch.tensor([x[3] for x in batch]),
            torch.tensor([x[4] for x in batch]),
            torch.tensor([x[5] for x in batch]),
            torch.tensor([x[6] for x in batch]),
            torch.tensor([x[7] for x in batch]),
            torch.stack([torch.from_numpy(x[8]) for x in batch]),
            torch.stack([torch.from_numpy(x[9]) for x in batch]),
        )
    else:
        return (
            torch.stack([torch.from_numpy(x[0]) for x in batch]),
            torch.tensor([x[1] for x in batch]),
            torch.tensor([x[2] for x in batch]),
            torch.tensor([x[3] for x in batch]),
            torch.tensor([x[4] for x in batch]),
            torch.tensor([x[5] for x in batch]),
            torch.tensor([x[6] for x in batch]),
            torch.tensor([x[7] for x in batch]),
            torch.stack([torch.from_numpy(x[8]) for x in batch]),
        )


class ChDataModule(pl.LightningDataModule):
    def __init__(self, datamodule_cfg, dataloader_cfg):
        super().__init__()
        self.cfg = datamodule_cfg
        self.dataloader_cfg = dataloader_cfg

    def setup(self, stage):
        snapshot_download(repo_id="climatehackai/climatehackai-2023", allow_patterns=["pv*"], repo_type="dataset", local_dir="data")
        # Load data
        if not os.path.exists("data/pv/all.parquet"):
            # Download data
            pv_data = []
            for year in [2020, 2021]:
                for month in range(1, 13):
                    pv_data.append(pd.read_parquet(f"data/pv/{year}/{month}.parquet"))
            pv_data = pd.concat(pv_data).drop("generation_wh", axis=1)
            pv_data.to_parquet("data/pv/all.parquet", engine="fastparquet")

        else:
            pv_data = pd.read_parquet("data/pv/all.parquet")

        if not os.path.exists("data/pv/train.parquet"):
            # Add day column
            pv_data['day'] = pv_data.index.get_level_values('timestamp').date
            days = pv_data['day'].unique()

            # Split into train and val with random days
            train_days = np.random.choice(days, int(len(days) * 0.8))
            train_pv = pv_data.loc[pv_data['day'].isin(train_days)]
            val_pv = pv_data.loc[~pv_data['day'].isin(train_days)]
            train_pv.drop("day", axis=1, inplace=True)
            val_pv.drop("day", axis=1, inplace=True)

            # Save
            train_pv.to_parquet("data/pv/train.parquet", engine="fastparquet")
            val_pv.to_parquet("data/pv/val.parquet", engine="fastparquet")
        else:
            train_pv = pd.read_parquet("data/pv/train.parquet")
            val_pv = pd.read_parquet("data/pv/val.parquet")

        meta_data = pd.read_csv("data/pv/metadata.csv", index_col=0)

        # hrv_data = xr.open_mfdataset(
        #     [f"data/satellite-hrv/{year}/{month}.zarr.zip" for year in [2020, 2021] for month in range(1, 13)],
        #     engine="zarr", chunks="auto", parallel=True
        # )

        with open("data/indices.json") as f:
            site_locations = {
                data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
                }
                for data_source, locations in json.load(f).items()
            }

        # Create datasets
        if "use_hrv" in self.cfg:
            use_hrv = self.cfg["use_hrv"]
        else:
            use_hrv = False

        self.train_dataset = ChDataset(train_pv, meta_data, site_locations, use_hrv)
        self.val_dataset = ChDataset(val_pv, meta_data, site_locations, use_hrv)

    def toggle_train_hrv(self):
        self.train_dataset.toggle_hrv()

    def toggle_val_hrv(self):
        self.val_dataset.toggle_hrv()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_cfg, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_cfg, shuffle=False, collate_fn=collate_fn)