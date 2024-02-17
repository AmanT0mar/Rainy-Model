import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
from sklearn.base import BaseEstimator, TransformerMixin

def weather_api_response(city_name):
    '''Getting data from weather API'''

    city_details = {
            "Chennai" : [13.08, 80.27],
            "Mayiladuthurai" : [11.10, 79.65],
            "Thoothukudi" : [8.76, 78.13],
            "Nagercoil" : [8.18, 77.41]
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
    date_three_days_ago = datetime.today() - timedelta(days=3)
    params = {
        "latitude": city_lat,
        "longitude": city_long,
        "start_date": date_three_days_ago.strftime('%Y-%m-%d'),
        "end_date": date_three_days_ago.strftime('%Y-%m-%d'),
        "daily": ["temperature_2m_max", "temperature_2m_min","rain_sum", "precipitation_hours",
                  "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "et0_fao_evapotranspiration"],
        "timezone": "GMT"
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
    except:
        print("Weather API request error")

    response = responses[0]
    daily = response.Daily()
    daily_data = {"date": date_three_days_ago.strftime('%Y-%m-%d'),
                  "city_name" : city_name
                  }
    daily_data["temperature_2m_max"] = daily.Variables(0).ValuesAsNumpy()
    daily_data["temperature_2m_min"] = daily.Variables(1).ValuesAsNumpy()
    daily_data["rain_sum"] = daily.Variables(2).ValuesAsNumpy()
    daily_data["precipitation_hours"] = daily.Variables(3).ValuesAsNumpy()
    daily_data["wind_speed_10m_max"] = daily.Variables(4).ValuesAsNumpy()
    daily_data["wind_gusts_10m_max"] = daily.Variables(5).ValuesAsNumpy()
    daily_data["wind_direction_10m_dominant"] = daily.Variables(6).ValuesAsNumpy()
    daily_data["et0_fao_evapotranspiration"] = daily.Variables(7).ValuesAsNumpy()
    daily_df = pd.DataFrame(data = daily_data)

    return daily_df

def date_encoder(df):
    '''Date Encoder---works for now'''

    df_encode = df.copy()
    for i in df_encode['day_of_year']:
        df_encode['day_of_year_sin'] = np.sin(2 * np.pi * i / 366)
        df_encode['day_of_year_cos'] = np.cos(2 * np.pi * i / 366)
    df = pd.concat([df['city_name'], df_encode[['day_of_year_sin','day_of_year_cos']],
                df[['temperature_2m_max','temperature_2m_min',
                'rain_sum','precipitation_hours','wind_speed_10m_max','wind_gusts_10m_max',
                'wind_direction_10m_dominant','et0_fao_evapotranspiration']]], axis=1)
    return df

def model_prediction(city_name):
    '''Model Prediction---shit model-shit data-shit processing'''

    predict_df = weather_api_response(city_name)

    ####Loading Model
    try:
        standard_scaler = joblib.load("D:\Final Project\Model-API\MLmodel\StandardScaler_1.pkl")
        city_encoder = joblib.load("D:\Final Project\Model-API\MLmodel\CityLabelEncoder_1.pkl")
        model = joblib.load("D:\Final Project\Model-API\MLmodel\MLR_1.pkl")
    except:
        print("Error unpickling models")
    ####Encoding Date
    try:
        predict_df['date'] = pd.to_datetime(predict_df['date'], format='%Y-%m-%d')
        predict_df['day_of_year'] = predict_df['date'].dt.dayofyear
        predict_df = date_encoder(predict_df)
    except:
        print("Error encoding date")
    ####Encoding City
    try:
        predict_df['city_name'] = city_encoder.transform(predict_df['city_name'])
    except:
        print("Error encoding cityname")
    ####Feature Scaling(Standard Scaler)
    try:
        numerical_columns = ["city_name","temperature_2m_max", "temperature_2m_min", "rain_sum",
                        "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max",
                        "wind_direction_10m_dominant","et0_fao_evapotranspiration"]
        predict_df[numerical_columns] = standard_scaler.transform(predict_df[numerical_columns])
    except:
        print("Error scaling features")
    ####Rainfall amount prediction
    predict_df = pd.concat([predict_df[['day_of_year_sin','day_of_year_cos']],
                        predict_df[["city_name","temperature_2m_max", "temperature_2m_min",
                         "rain_sum","precipitation_hours", "wind_speed_10m_max",
                        "wind_gusts_10m_max","wind_direction_10m_dominant","et0_fao_evapotranspiration"]]],
                        axis=1)
    try:
        prediction = model.predict(predict_df)
    except:
        print("Error predicting")

    return prediction[0]