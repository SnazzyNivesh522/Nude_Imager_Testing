import os
import requests
import json
from sklearn.metrics import confusion_matrix
import numpy as np

ENDPOINTS = {
    "nsfwjs": "http://localhost:3000/classify/binary",
    "falconsai": "http://localhost:5002/classify/binary",
}

DATASET_PATH = "dataset"


def get_predictions(image_path, endpoint_url):
    """Sends an image to the specified endpoint and returns the predicted class."""
    try:
        if "3000" in endpoint_url:
            file_key = "image"
        else:
            file_key = "file"

        with open(image_path, "rb") as f:
            files = {file_key: f}
            response = requests.post(endpoint_url, files=files)
            response.raise_for_status()
            data = response.json()
            return data.get("classification")
    except requests.exceptions.RequestException as e:
        print(f"Error calling {endpoint_url} with image {image_path}: {e}")
        return None


def evaluate_endpoint(endpoint_name, endpoint_url, dataset_path):
    """Evaluates a single endpoint and returns true and predicted labels."""
    true_labels = []
    predicted_labels = []

    print(f"--- Evaluating {endpoint_name} endpoint ---")

    for class_name in ["normal", "nsfw"]:
        class_path = os.path.join(dataset_path, class_name)
        counter = 0
        for filename in os.listdir(class_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                counter += 1
                image_path = os.path.join(class_path, filename)

                # Get prediction from the model
                prediction = get_predictions(image_path, endpoint_url)

                if prediction:
                    true_labels.append(class_name)
                    predicted_labels.append(prediction)
                    print(
                        f"Image: {filename}, True: {class_name}, Predicted: {prediction}"
                    )
            if counter == 1000:
                break

    return true_labels, predicted_labels


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Prints a formatted confusion matrix."""
    print(f"\n{title}:")

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print(f"{'':>12s} | {'Predicted Normal':>18s} | {'Predicted NSFW':>16s}")
    print("-" * 55)
    print(
        f"{'Actual Normal':>12s} | {cm[0, 0]:>18d} ({cm_normalized[0, 0]:.2f}) | {cm[0, 1]:>16d} ({cm_normalized[0, 1]:.2f})"
    )
    print(
        f"{'Actual NSFW':>12s} | {cm[1, 0]:>18d} ({cm_normalized[1, 0]:.2f}) | {cm[1, 1]:>16d} ({cm_normalized[1, 1]:.2f})"
    )

    print("\n")


if __name__ == "__main__":
    for name, url in ENDPOINTS.items():
        true_labels, predicted_labels = evaluate_endpoint(name, url, DATASET_PATH)

        if true_labels and predicted_labels:
            labels = ["normal", "nsfw"]
            cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

            plot_confusion_matrix(
                cm, labels, title=f"Confusion Matrix for {name} Model"
            )

            tn, fp, fn, tp = cm.ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall (Sensitivity): {recall:.4f}")
        else:
            print(f"No predictions were made for {name}. Check endpoint and dataset.")
