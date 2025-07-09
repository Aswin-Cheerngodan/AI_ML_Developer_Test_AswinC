from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

from classify import TrendClassifier  # Replace with actual import if in different file

app = FastAPI()

# Mount static and template folders
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    classifier = TrendClassifier(file)
    trend = classifier.classify()

    return templates.TemplateResponse("index.html", {"request": request, "result": trend})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
