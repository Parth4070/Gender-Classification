from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.predict import predict_image


app = FastAPI(
    title="Face Gender Classification API",
    description="ResNet18-based image classification API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Face Gender Classification API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(BytesIO(image_bytes))
    prediction, confidence = predict_image(image)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }