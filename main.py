import pickle
import uvicorn
from fastapi import FastAPI
import numpy as np

with open("model.pkl","rb")as f:
    model = pickle.load(f)

app = FastAPI()

from pydantic import BaseModel

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

@app.post("/iris_predict")

def predict(data : request_body):
    test_data = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
    ]]
    species = ["Iris setosa", "Iris virginica",  "Iris versicolor"]
    pred = model.predict(test_data)[0]

    return {"Pred":species[pred]}
