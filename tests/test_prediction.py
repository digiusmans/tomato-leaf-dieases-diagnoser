# from PIL import Image
# from src.predict import predict_disease

# image = Image.open("test_images/test_leaf.jpg")

# disease, confidence = predict_disease(image)

# print("Prediction:", disease)
# print("Confidence:", f"{confidence:.2f}%")



from PIL import Image
from predict import predict_disease
import os

test_folder = "../test_images"

for file_name in os.listdir(test_folder):
    file_path = os.path.join(test_folder, file_name)

    image = Image.open(file_path)

    results = predict_disease(image)

    print(f"\nTesting: {file_name}")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['label']} -> {result['confidence']:.2f}%")