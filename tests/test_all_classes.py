
from pathlib import Path

from PIL import Image
from src.predict import predict_disease

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".jfif"
}


def test_all_classes():
    assert TEST_IMAGES_DIR.exists(), "Test images folder not found."

    image_files = sorted(
        path for path in TEST_IMAGES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    assert image_files, "No test images found."

    for image_path in image_files:
        with Image.open(image_path) as image:
            result = predict_disease(image)

        assert isinstance(result, dict)
        assert "valid" in result

        if result["valid"]:
            assert "predictions" in result

            for prediction in result["predictions"]:
                assert "label" in prediction
                assert "confidence" in prediction
