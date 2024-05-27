import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from modular.modular.model_builder import LSTMModel
from modular.modular.data_setup import split_dataframe, create_sequence
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

device = 'cpu'
TF_ENABLE_ONEDNN_OPTS=0

def weather_api_response(city_name: str) -> pd.DataFrame:
    """
    API call to get data for model prediction

    Args:
        city_name: name of city for requesting data
    """

    city_details = {
        "Chennai" : [13.08, 80.27],
        "Mayiladuthurai" : [11.10, 79.65],
        "Thoothukudi" : [8.76, 78.13],
        "Nagercoil" : [8.18, 77.41],
        "Thiruvananthapuram": [8.53, 76.94],
        "Kollam": [8.89, 76.61],
        "Kochi": [9.95, 76.26],
        "Kozhikode": [11.26, 75.77],
        "Kannur": [11.87, 75.37],
        "Visakhapatnam": [17.69, 83.23],
        "Nellore": [14.44, 79.98],
        "Mangaluru": [13.01, 74.92],
        "Udupi": [13.34, 74.74],
        "Mumbai": [19.09, 72.84],
        "Daman": [20.40, 72.83],
        "Alappuzha": [9.50, 76.34],
        "Kakinada": [16.98, 82.25]
    }
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
    except:
        print("Weather API client error")
    try:
        city_lat = city_details[city_name][0]
        city_long = city_details[city_name][1]
    except KeyError:
        print("City prediction not available")

    url = "https://archive-api.open-meteo.com/v1/archive"
    current_date = datetime.today() - timedelta(days=3)
    lookback_date = datetime.today() - timedelta(days=182)
    params = {
        "latitude": city_lat,
        "longitude": city_long,
        "start_date": lookback_date.strftime('%Y-%m-%d'),
        "end_date": current_date.strftime('%Y-%m-%d'),
        "daily": ["precipitation_sum"],
        "timezone": "Asia/Kolkata"
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
    except:
        print("Weather API request error")

    response = responses[0]
    daily = response.Daily()
    daily_data = {"date": pd.date_range(
                            start = pd.to_datetime(daily.Time(), unit = "s"),
                            end = pd.to_datetime(daily.TimeEnd(), unit = "s"),
                            freq = pd.Timedelta(seconds = daily.Interval()),
                            inclusive = "left"
                        ).strftime('%Y-%m-%d'),
                "city_name" : city_name
                }

    daily_data["precipitation_sum"] = daily.Variables(0).ValuesAsNumpy()

    daily_df = pd.DataFrame(data = daily_data)

    return daily_df


def load_model(city_name: str) -> LSTMModel:
    """
    Loads trained model for prediction.

    Args:
    city_name: name of the city to load particular model.
    """

    current_dir = Path.cwd()

    model_path = current_dir / "models"

    model_name = f"{city_name}_model.h5"

    model_save_path = model_path / model_name

    if not model_save_path.exists():
        raise FileNotFoundError(f"Model file for {city_name} not found at {model_save_path}")

    model = tf.keras.models.load_model(model_save_path)

    return model



def model_prediction(model: nn.Module, date_diff: int, X: torch.Tensor, df: pd.DataFrame, scaler: MinMaxScaler) -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Makes prediction using the given model.

    Args:
        model: The neural network to use for predictions.
        date_diff: To calculate the prediction date.
        X: The input data tensor for prediction.
        df: The Dataframe to which prediction result will be append.
        scaler: The scaler object used to unscale prediction result.
    """

    predicted_value = model.predict(X)

    date = datetime.now() + timedelta(days=(date_diff - 3))

    if scaler is not None:
        df.loc[len(df)] = [date, scaler.inverse_transform([[predicted_value.item()]])[0][0]]
    else:
        df.loc[len(df)] = [date, predicted_value.item()]

    X = np.roll(X, -1)
    X[-1] = predicted_value

    return df, X


def daily_prediction(model: nn.Module, date: datetime, city_name: str, scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Predicts daily precipitation till given date for the specified city using a neural network model.

    Args:
        model: The neural network to use for predictions.
        date: The date for which precipitation predicition in required.
        city_name: The name of the city for which predictions are made.
        scaler: The scaler object used for scaling  input and output data.
    """

    rainfall_df = pd.DataFrame(data=None, columns=['date', 'precipitation_sum'])

    df = weather_api_response(city_name=city_name)

    current_date = datetime.now() - timedelta(days=3)

    date_diff = (date - current_date).days + 1

    city_weather_df = split_dataframe(df=df, city_name=city_name, scaler=scaler)

    X = []
    for i in range(len(city_weather_df) - 179):
        X.append(city_weather_df.precipitation_sum.iloc[i:i+180])

    X = np.array(X)

    for i in range(1, date_diff+1):

        rainfall_df, X = model_prediction(model=model, date_diff=i, X=X, df=rainfall_df, scaler=scaler)

    return rainfall_df


def prediction(city_name: str, date: str):
    """
    Loads trained model and preprocesses data before making precipitation predictions for a given city on a specific date.

    Args:
        city_name: The name of the city for which predictions are made.
        date: The date till which precipitation predictions is required.
    """

    try:
        scaler = joblib.load("models/scaler.pkl")
        model = load_model(city_name=city_name)
    except:
        print("Error unpickling")

    try:
        prediction_df = daily_prediction(model=model, date=date, city_name=city_name, scaler=scaler)
    except:
        print("Error predicting")

    return prediction_df


def disaster_prediction(city_name: str, date: str) -> dict[str, bool]:
    """
    Predicts the likelihood of flood and drought disaster for a given city on a specific date.

    Args:
        city_name: The name of the city for which predictions are made.
        date: The date of which disaster prediction is required.
    """

    date = datetime.strptime(date, "%Y-%m-%d")
    prediction_df = prediction(city_name=city_name, date=date)

    results = {"flood" : False, "drought" : False}

    flood_cnt = sum(prediction_df['precipitation_sum'] > 25)
    drought_cnt = sum(prediction_df['precipitation_sum'] < 1)

    if flood_cnt >= len(prediction_df) * 0.8:
        results['flood'] = True

    if drought_cnt >= len(prediction_df) * 0.99:
        results['drought'] = False

    return results
