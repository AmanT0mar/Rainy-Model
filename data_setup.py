import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
import numpy as np
from copy import deepcopy

LOOKBACK = 60
BATCH_SIZE = 32


class RainfallDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def split_dataframe(df: pd.DataFrame, city_name: str) -> pd.DataFrame:

    city_df = deepcopy(df[df['city_name'] == city_name])

    city_df.drop(['city_name'], axis=1, inplace=True)

    city_df = city_df.set_index('date')

    return city_df


def create_lag_dataframe(df: pd.DataFrame,  scaler, n_steps: int = LOOKBACK) -> pd.DataFrame:

    df = deepcopy(df)

    for i in range(1, n_steps + 1):
        new_shifted_df = df['precipitation_sum'].shift(i)
        new_scaled_df = scaler.transform(np.expand_dims(new_shifted_df.values, axis=1))
        df[f"precipitation_sum(t-{i})"] = new_scaled_df

    df.dropna(inplace=True)

    return df


def create_X_y(df: pd.DataFrame):

    df_as_np = df.to_numpy()

    X = df_as_np[:, 1:]
    y = df_as_np[:, 0]

    X = deepcopy(np.flip(X, axis=1))

    return X, y

def split_dataframe(df: pd.DataFrame, city_name: str) -> pd.DataFrame:

    city_df = deepcopy(df[df['city_name'] == city_name])

    city_df.drop(['city_name'], axis=1, inplace=True)

    city_df = city_df.set_index('date')

    return city_df


def create_lag_dataframe(df: pd.DataFrame,  scaler, n_steps: int = LOOKBACK) -> pd.DataFrame:

    df = deepcopy(df)

    for i in range(1, n_steps + 1):
        new_shifted_df = df['precipitation_sum'].shift(i)
        new_scaled_df = scaler.transform(np.expand_dims(new_shifted_df.values, axis=1))
        df[f"precipitation_sum(t-{i})"] = new_scaled_df

    df.dropna(inplace=True)

    return df


def create_X_y(df: pd.DataFrame):

    df_as_np = df.to_numpy()

    X = df_as_np[:, 1:]
    y = df_as_np[:, 0]

    X = deepcopy(np.flip(X, axis=1))

    return X, y

def create_dataloader(X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor,
                    batch_size: int = BATCH_SIZE) -> DataLoader:

    train_dataset = RainfallDataset(X_train, y_train)
    test_dataset = RainfallDataset(X_test, y_test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader