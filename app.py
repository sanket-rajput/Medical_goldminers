from fastapi import FastAPI
from pydantic import BaseModel
from services.analyzer import analyze

app = FastAPI()

class SymptomInput(BaseModel):
    symptoms: list[str]

@app.post("/analyze")
def analyze_symptoms(data: SymptomInput):
    result = analyze(data.symptoms)
    return {"results": result}