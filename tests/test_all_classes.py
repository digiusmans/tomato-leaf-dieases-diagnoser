from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
SRC_DIR = ROOT_DIR / "src"
TEST_IMAGES_DIR = ROOT_DIR / "test_images"

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from predict import predict_disease  # noqa: E402

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}


def main() -> None:
    if not TEST_IMAGES_DIR.exists():
        print(f"Test images folder not found: {TEST_IMAGES_DIR}")
        sys.exit(1)

    image_files = sorted(
        [path for path in TEST_IMAGES_DIR.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    if not image_files:
        print(f"No test images found in: {TEST_IMAGES_DIR}")
        sys.exit(1)

    for image_path in image_files:
        print(f"Testing: {image_path.name}\n")
        try:
            with Image.open(image_path) as image:
                predictions = predict_disease(image)

            for index, prediction in enumerate(predictions, start=1):
                print(f"{index}. {prediction['label']} -> {prediction['confidence']:.2f}%")
            print()

        except Exception as exc:
            print(f"Failed to process {image_path.name}: {exc}\n")


if __name__ == "__main__":
    main()