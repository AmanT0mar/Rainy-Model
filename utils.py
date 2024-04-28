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
from sklearn.base import BaseEstimator, TransformerMixin
from model_builder import LSTMModel
from data_setup import split_dataframe, create_lag_dataframe
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

device = 'cpu'

def weather_api_response(city_name: str):
    """
        API call to get data for model prediction
        -----------------------------------------
        Returns pandas.DataFrame
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
    lookback_date = datetime.today() - timedelta(days=62)
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


def load_model(city: str):

    model_path = Path("models")

    model_name = f"{city}_model.pth"

    model_save_path = model_path / model_name

    model_state_dict = torch.load(f=model_save_path)

    model = LSTMModel(input_size=1, hidden_size=8, num_layers=2, output_size=1)

    model.load_state_dict(model_state_dict)

    return model



def model_prediction(model: nn.Module, date_diff: int, X: torch.Tensor, df: pd.DataFrame, scaler: MinMaxScaler):

    prediction = model(X.to(device))

    date = datetime.now() - timedelta(days=3) + timedelta(days=date_diff)

    df.loc[len(df)] = [date, scaler.inverse_transform([[prediction.item()]])[0][0]]

    X_new = torch.cat((X[:, 1:].to(device), prediction.unsqueeze(0).to(device)), dim=1)

    return df, X_new


def daily_prediction(model: nn.Module, date: datetime, city: str, scaler: MinMaxScaler) -> pd.DataFrame:

    rainfall_df = pd.DataFrame(data=None, columns=['date', 'precipitation_sum'])

    df = weather_api_response(city_name=city)

    current_date = datetime.now() - timedelta(days=3)

    date_diff = (date - current_date).days + 1

    city_weather_df = split_dataframe(df=df, city_name=city)

    city_lag_df = create_lag_dataframe(df=city_weather_df, scaler=scaler, n_steps=59)

    df_as_np = city_lag_df.to_numpy()

    X = deepcopy(np.flip(df_as_np, axis=1))

    X = X.reshape((-1, 60, 1))
    X = torch.tensor(X, dtype=torch.float)

    for i in range(1, date_diff+1):

        rainfall_df, X = model_prediction(model=model, date_diff=i, X=X, df=rainfall_df, scaler=scaler)

    return rainfall_df


def prediction(city_name: str, date: str):
    """
        Loads trained models and preprocesses data before model prediction
        -----------------------------------------
        Returns boolean
    """

    df = weather_api_response(city_name)
    predict_df = pd.DataFrame([df.iloc[-1].copy()], columns=df.columns)
    #predict_df = df.iloc[-1, :].copy().to_frame()
    ####Loading Model
    try:
        scaler = joblib.load("models\minmaxscaler.pkl")
        model = load_model(city=city_name)
    except:
        print("Error unpickling")

    try:
        prediction_df = daily_prediction(model=model, date=date, city=city_name, scaler=scaler)
    except:
        print("Error predicting")

    return prediction_df

def disaster_prediction(city_name: str, date: str) -> dict[str, bool]:

    date = datetime.strptime(date, "%Y-%m-%d")
    prediction_df = prediction(city_name=city_name, date=date)

    results = {"flood" : False, "drought" : False}

    flood_cnt = 0
    for i in range(len(prediction_df)):
        if prediction_df['precipitation_sum'][i] > 10:
            flood_cnt += 1

    drought_cnt = 0
    for i in range(len(prediction_df)):
        if prediction_df['precipitation_sum'][i] < 1:
            drought_cnt += 1

    if flood_cnt >= len(prediction_df) * 0.8:
        results['flood'] = True

    if drought_cnt <= len(prediction_df) * 0.9:
        results['drought'] = True


    return results
