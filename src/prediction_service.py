from src.predict import (
    predict_disease,
    get_confidence_level,
    get_disease_info
)


def process_prediction(image):

    result = predict_disease(image)

    if not result["valid"]:
        return result

    top_prediction = result["predictions"][0]

    disease_name = top_prediction["label"]
    confidence = top_prediction["confidence"]

    return {
        "valid": True,
        "disease": disease_name,
        "confidence": confidence,
        "confidence_level": get_confidence_level(confidence),
        "predictions": result["predictions"],
        "disease_info": get_disease_info(disease_name)
    }