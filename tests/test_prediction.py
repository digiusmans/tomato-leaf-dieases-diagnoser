# from PIL import Image
# from src.predict import predict_disease

# image = Image.open("test_images/test_leaf.jpg")

# disease, confidence = predict_disease(image)

# print("Prediction:", disease)
# print("Confidence:", f"{confidence:.2f}%")



from PIL import Image
from src.predict import predict_disease
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES = PROJECT_ROOT / "test_images"


def test_prediction():
    image_files = [
        path for path in TEST_IMAGES.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    assert image_files, "No test images found."

    for file_path in image_files:
        with Image.open(file_path) as image:
            result = predict_disease(image)

        assert isinstance(result, dict)
        assert "valid" in result

        if result["valid"]:
            assert "predictions" in result

            for prediction in result["predictions"]:
                assert "label" in prediction
                assert "confidence" in prediction