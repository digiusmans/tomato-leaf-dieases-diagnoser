from io import BytesIO

from fastapi import UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from src.prediction_service import process_prediction
from fastapi import FastAPI

app = FastAPI(
    title="Tomato Disease Detection API",
    description="API for detecting tomato leaf diseases using deep learning.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to Tomato Disease Detection API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
@app.post("/predict")
async def predict_tomato_disease(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported."
        )

    try:
        image_data = await file.read(MAX_FILE_SIZE + 1)

        if len(image_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 5 MB."
            )

        image = Image.open(BytesIO(image_data))
        image.load()

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    finally:
        await file.close()

    return process_prediction(image)