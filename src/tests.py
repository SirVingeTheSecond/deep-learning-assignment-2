import os
import sys
import torch
from cnn import CNN
from data import load_data
from config import config
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def evaluateModel(model: CNN, x_test, y_test):
    model.eval()
    device = config.get("device", "cpu")

    Xte_t = torch.from_numpy(x_test).to(torch.float32).to(device)
    yte_t = torch.from_numpy(y_test).to(torch.long).to(device)

    with torch.no_grad():
        class_names = ["CNV", "DME", "Drusen", "Normal"]
        classes = len(class_names)
        
        outputs = model(Xte_t)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == yte_t).sum().item()
        total = yte_t.size(0)
        accuracy = correct / total * 100
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, predicted.cpu().numpy())
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_labels = [c[:] for c in class_names]
        plt.xticks(range(classes), tick_labels, rotation=45, ha='right')
        plt.yticks(range(classes), tick_labels)

        # Add text annotations
        for i in range(classes):
            for j in range(classes):
                plt.text(j, i, str(round(cm[i, j], 2)), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        plt.close()

        # Conditional probability P(True = i | Predicted = j)
        cm = confusion_matrix(y_test, predicted.cpu().numpy())

        # Column-normalized confusion matrix (prediction-conditioned)
        col_sums = cm.sum(axis=0, keepdims=True)
        conditional_prob = cm / col_sums

        print("\nConditional probability P(True | Predicted):")
        for i, pred_class in enumerate(class_names):
            for j, true_class in enumerate(class_names):
                print(f"P(True={true_class} | Pred={pred_class}) = {conditional_prob[j, i]:.3f}")
            print()

def evaulateModel(weights_path: str):
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    device = config.get("device", "cpu")

    device = config.get("device", "cpu")

    model = CNN(num_classes=config.get("num_classes", 4))
    model.to(device)

    image_size = config.get("image_size", 64)
    dummy = torch.randn(1, 1, image_size, image_size).to(device)
    model(dummy) # initializes Lazy layers

    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    evaluateModel(model, x_test, y_test)



if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: python main.py <path_to_model_weights>")
        sys.exit(1)

    if (not sys.argv[1].endswith('.pth')):
        print("Please provide a valid model weights file with .pth extension.")
        sys.exit(1)

    if (not os.path.isfile(sys.argv[1])):
        print(f"File {sys.argv[1]} does not exist.")
        sys.exit(1)

    evaulateModel(sys.argv[1])