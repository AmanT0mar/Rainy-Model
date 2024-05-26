from utils import disaster_prediction
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow requests from specific origins if needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/city_list")
def read_model():
    return {"statusCode" : 200,
            "city_list" : ["Chennai","Mayiladuthurai","Thoothukudi","Nagercoil","Thiruvananthapuram","Kollam","Kochi","Kozhikode","Kannur","Visakhapatnam","Nellore","Mangaluru","Udupi","Mumbai","Daman","Alappuzha","Kakinada"]}

@app.get("/rainfall/{city_name}/{date}")
def city_data(city_name: str, date: str):
    prediction = disaster_prediction(city_name=city_name, date=date)
    if prediction is None:
        return {'statusCode' : 500,
                'data' : None}
    return {'statusCode' : 200,
            'data' : prediction}