from utils import model_prediction
from fastapi import FastAPI

app = FastAPI()

@app.get("/city_list")
def read_model():
    return {"statusCode" : 200,
            "city_list" : ["Chennai","Mayiladuthurai","Thoothukudi","Nagercoil","Thiruvananthapuram","Kollam","Kochi","Kozhikode","Kannur","Visakhapatnam","Nellore","Mangaluru","Udupi","Mumbai","Daman","Alappuzha","Kakinada"]}

@app.get("/rainfall/{city_name}")
def city_data(city_name):
    prediction = model_prediction(city_name=city_name)
    if prediction is None:
        return {'statusCode' : 500,
                'data' : []}
    return {'statusCode' : 200,
            'data' : [prediction]}