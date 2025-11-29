import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def evaluate_model(model, x_test, y_test, class_names, device):
    model.eval()

    x_test_t = torch.from_numpy(x_test).to(torch.float32).to(device)
    y_test_t = torch.from_numpy(y_test).to(torch.long).to(device)

    with torch.no_grad():
        outputs = model(x_test_t)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test_t).sum().item()
        total = y_test_t.size(0)
        accuracy = correct / total

        cm = confusion_matrix(y_test, predicted.cpu().numpy())

        return {
            "accuracy": accuracy,
            "predictions": predicted.cpu().numpy(),
            "confusion_matrix": cm,
        }


def display_confusion_matrix(cm, class_names):
    num_classes = len(class_names)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_labels = [c[:] for c in class_names]
    plt.xticks(range(num_classes), tick_labels, rotation=45, ha='right')
    plt.yticks(range(num_classes), tick_labels)

    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(round(cm[i, j], 2)), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    plt.close()


def print_conditional_probabilities(cm, class_names):
    # Column-normalized confusion matrix (prediction-conditioned)
    col_sums = cm.sum(axis=0, keepdims=True)
    conditional_prob = cm / col_sums

    print("\nConditional probability P(True | Predicted):")
    for i, pred_class in enumerate(class_names):
        for j, true_class in enumerate(class_names):
            print(f"P(True={true_class} | Pred={pred_class}) = {conditional_prob[j, i]:.3f}")
        print()