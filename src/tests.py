import os
import sys
import torch

from models import CNN, ConfigurableCNN
from data import load_data
from config import config
from evaluation import evaluate_model, display_confusion_matrix, print_conditional_probabilities
from utils import get_device


def evaluate_checkpoint(weights_path: str):
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    device = get_device(config.get("device", "cpu"))

    checkpoint = torch.load(weights_path, map_location=device)

    model = CNN(num_classes=config.get("num_classes", 4))
    model.to(device)

    image_size = config.get("image_size", 64)
    dummy = torch.randn(1, 1, image_size, image_size).to(device)
    model(dummy)

    try:
        model.load_state_dict(checkpoint.get("model_state_dict"))
    except RuntimeError:
        print("Warning: Standard CNN architecture doesn't match checkpoint.")
        print("Checkpoint may be from ConfigurableCNN. Attempting to load...")
        model = ConfigurableCNN(num_classes=config.get("num_classes", 4))
        model.to(device)
        model.load_state_dict(checkpoint.get("model_state_dict"))

    class_names = config.get("class_names", ["CNV", "DME", "Drusen", "Normal"])

    results = evaluate_model(model, x_test, y_test, class_names, device)

    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}%")

    display_confusion_matrix(results['confusion_matrix'], class_names)
    print_conditional_probabilities(results['confusion_matrix'], class_names)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tests.py <path_to_model_weights>")
        sys.exit(1)

    if not sys.argv[1].endswith('.pth') and not sys.argv[1].endswith('.pt'):
        print("Please provide a valid model weights file with .pth or .pt extension.")
        sys.exit(1)

    if not os.path.isfile(sys.argv[1]):
        print(f"File {sys.argv[1]} does not exist.")
        sys.exit(1)

    evaluate_checkpoint(sys.argv[1])
