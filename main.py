from fastapi import FastAPI
from pydantic import BaseModel
import pickle

#LOAD THE MODEL
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Creating app
app = FastAPI()


class HouseFreatures(BaseModel):
    square_footage: int
    number_of_bedrooms: int
    number_of_bathrooms: int
    zip_code: int
    price: int = None  # Optional, as it will be predicted

@app.post("/predict")
def predict_price(features: HouseFreatures):
    input_data = [[
        features.square_footage,
        features.number_of_bedrooms,
        features.number_of_bathrooms,
        features.zip_code
    ]]
    prediction = model.predict(input_data)
    return {"predicted_price": prediction[0]}
