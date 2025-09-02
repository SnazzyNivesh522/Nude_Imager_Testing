import requests
import os
import json

API_URL = "http://localhost:9080/infer"
DATA_PATH = "dataset"
NSFW_LABELS = [
    "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
]
SCORE_THRESHOLD = 0.5


def get_model_prediction(image_path):
    try:
        with open(image_path, "rb") as image_file:
            files = {"f1": (os.path.basename(image_path), image_file, "image/jpeg")}
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the API: {e}")
        return None


def is_nsfw_prediction(prediction_data):
    """
    Checks if the model's prediction contains an NSFW label above the threshold.
    """
    if (
        prediction_data
        and "prediction" in prediction_data
        and prediction_data["prediction"]
    ):
        for item in prediction_data["prediction"][0]:
            if "class" in item and "score" in item:
                if item["class"] in NSFW_LABELS and item["score"] > SCORE_THRESHOLD:
                    return True
    return False


def evaluate_model():
    """
    Evaluates the model and prints the confusion matrix.
    """
    confusion_matrix = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }

    # Process NSFW folder
    nsfw_path = os.path.join(DATA_PATH, "nsfw")
    if not os.path.exists(nsfw_path):
        print("NSFW folder not found. Please create it and add images.")
        return

    print("--- Evaluating NSFW images... ---")
    count = 0
    for filename in os.listdir(nsfw_path):
        image_path = os.path.join(nsfw_path, filename)
        if os.path.isfile(image_path):
            count += 1
            prediction = get_model_prediction(image_path)
            if is_nsfw_prediction(prediction):
                confusion_matrix["true_positive"] += 1
            else:
                confusion_matrix["false_negative"] += 1
        if count == 1000:
            break

    # Process Normal folder
    normal_path = os.path.join(DATA_PATH, "normal")
    if not os.path.exists(normal_path):
        print("Normal folder not found. Please create it and add images.")
        return

    print("\n--- Evaluating Normal images... ---")
    count = 0
    for filename in os.listdir(normal_path):
        image_path = os.path.join(normal_path, filename)
        if os.path.isfile(image_path):
            count += 1
            prediction = get_model_prediction(image_path)
            if is_nsfw_prediction(prediction):
                confusion_matrix["false_positive"] += 1
            else:
                confusion_matrix["true_negative"] += 1
        if count == 1000:
            break

    # Print the confusion matrix and metrics
    print("\n--- Results ---")
    print(
        f"Number of NSFW images processed: {confusion_matrix['true_positive'] + confusion_matrix['false_negative']}"
    )
    print(
        f"Number of Normal images processed: {confusion_matrix['true_negative'] + confusion_matrix['false_positive']}"
    )

    print("\nConfusion Matrix:")
    print("---------------------------------------")
    print(f"| {'':<15} | {'Predicted Normal':<15} | {'Predicted NSFW':<15} |")
    print("---------------------------------------")
    print(
        f"| {'Actual Normal':<15} | {confusion_matrix['true_negative']:<15} | {confusion_matrix['false_positive']:<15} |"
    )
    print(
        f"| {'Actual NSFW':<15} | {confusion_matrix['false_negative']:<15} | {confusion_matrix['true_positive']:<15} |"
    )
    print("---------------------------------------")

    total_images = sum(confusion_matrix.values())
    if total_images > 0:
        accuracy = (
            confusion_matrix["true_positive"] + confusion_matrix["true_negative"]
        ) / total_images
        print(f"\nAccuracy: {accuracy:.2f}")

    if (confusion_matrix["true_positive"] + confusion_matrix["false_positive"]) > 0:
        precision = confusion_matrix["true_positive"] / (
            confusion_matrix["true_positive"] + confusion_matrix["false_positive"]
        )
        print(f"Precision: {precision:.2f}")

    if (confusion_matrix["true_positive"] + confusion_matrix["false_negative"]) > 0:
        recall = confusion_matrix["true_positive"] / (
            confusion_matrix["true_positive"] + confusion_matrix["false_negative"]
        )
        print(f"Recall: {recall:.2f}")


if __name__ == "__main__":
    evaluate_model()
