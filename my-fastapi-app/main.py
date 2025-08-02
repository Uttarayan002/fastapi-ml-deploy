from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained model
model = joblib.load("iris_model.pkl")

#CREATE FASTAPI APP
app = FastAPI()

# Define request body structure
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define a prediction route
@app.post("/predict")
def predict(data: IrisRequest):
    # Convert input to numpy array
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return {"prediction": int(prediction)}
