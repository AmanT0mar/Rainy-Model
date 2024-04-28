from utils import disaster_prediction
from fastapi import FastAPI

app = FastAPI()

@app.get("/city_list")
def read_model():
    return {"statusCode" : 200,
            "city_list" : ["Chennai","Mayiladuthurai","Thoothukudi","Nagercoil","Thiruvananthapuram","Kollam","Kochi","Kozhikode","Kannur","Visakhapatnam","Nellore","Mangaluru","Udupi","Mumbai","Daman","Alappuzha","Kakinada"]}

@app.get("/rainfall")
def city_data(city_name: str, date: str):
    prediction = disaster_prediction(city_name=city_name, date=date)
    if prediction is None:
        return {'statusCode' : 500,
                'data' : None}
    return {'statusCode' : 200,
            'data' : prediction}