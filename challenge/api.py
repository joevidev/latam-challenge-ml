import fastapi
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator
from .model import DelayModel
import pandas as pd
from typing import List

app = fastapi.FastAPI()
model = DelayModel()

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int
    Fecha_I: str
    Fecha_O: str
    
    @validator("TIPOVUELO")
    def validate_TIPOVUELO(cls, v):
        if v not in ["N", "I"]:
            raise ValueError("Invalid flight type")
        return v
    
    @validator("MES")
    def validate_MES(cls, v):
        if v not in range(1, 13):
            raise ValueError("Invalid month")
        return v

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return fastapi.responses.JSONResponse(
        status_code=400,
        content={"detail": exc.errors()}
    )

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

class FlightList(BaseModel):
    flights: List[FlightData]

@app.post("/predict", status_code=200)
async def post_predict(data: FlightList) -> dict:
    try:
        df = pd.DataFrame([flight.dict() for flight in data.flights])
        df.rename(columns={'Fecha_I': 'Fecha-I', 'Fecha_O': 'Fecha-O'}, inplace=True)
        features = model.preprocess(df)
        predictions = model.predict(features)
        return {"predict": predictions}
    except Exception as e:
        return {"error": "An error occurred during prediction.", "detail": str(e)}
