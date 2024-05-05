from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# Allowing CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading the model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.get("/")
async def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return {
        'companies': companies,
        'car_models': car_models,
        'years': year,
        'fuel_types': fuel_type
    }

@app.post("/predict")
async def predict(company: str = Form(...), 
                  car_model: str = Form(...), 
                  year: int = Form(...), 
                  fuel_type: str = Form(...), 
                  driven: float = Form(...)):

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                             data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    return {"prediction": round(prediction[0], 2)}
