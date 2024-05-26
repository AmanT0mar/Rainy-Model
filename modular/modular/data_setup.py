import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

LOOKBACK = 60
BATCH_SIZE = 32


class RainfallDataset(Dataset):
    """
        Creates custom Dataset.

    Args:
    X: tensor of independent features.
    y: tensor of dependent feature.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


def create_dataloader(X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor,
                    batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    """
        Creates train and test dataloaders.
    """

    train_dataset = RainfallDataset(X_train, y_train)
    test_dataset = RainfallDataset(X_test, y_test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader


def split_dataframe(df: pd.DataFrame, city_name: str, scaler: MinMaxScaler) -> pd.DataFrame:
    """
        Creates a dataframe for a particular city.
    """

    city_df = deepcopy(df[df['city_name'] == city_name])

    city_df.drop(['city_name'], axis=1, inplace=True)

    if scaler is not None:
        city_df['precipitation_sum'] = scaler.transform(city_df[['precipitation_sum']])

    city_df = city_df.set_index('date')

    return city_df


def create_sequence(df: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
        Creates sequences of sequence_length.
    """

    sequences = []
    targets = []

    for i in range(len(df) - sequence_length):
        sequences.append(df.precipitation_sum.iloc[i:i+sequence_length])
        targets.append(df.precipitation_sum.iloc[i+sequence_length])

    return np.array(sequences), np.array(targets)


def create_lag_dataframe(df: pd.DataFrame, n_steps: int = LOOKBACK) -> pd.DataFrame:
    """
        Creates lag features of n_steps.
    """

    df = deepcopy(df)

    for i in range(1, n_steps + 1):
        df[f"precipitation_sum(t-{i})"] = df['precipitation_sum'].shift(i)

    df.dropna(inplace=True)

    return df


def create_X_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
        Splits dataframe into X and y ndarray.
    """

    df_as_np = df.to_numpy()

    X = df_as_np[:, 1:]
    y = df_as_np[:, 0]

    X = deepcopy(np.flip(X, axis=1))

    return X, y


