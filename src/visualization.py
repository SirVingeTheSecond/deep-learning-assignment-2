import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch


def visualize_dataset(train, val, test, class_names, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    rows = 4

    examples = {i: [] for i in range(len(class_names))}
    for i, label in enumerate(train.labels):
        label_idx = int(label[0])
        if len(examples[label_idx]) < rows:
            examples[label_idx].append(i)
        if all(len(examples[j]) == rows for j in range(len(class_names))):
            break

    plt.figure(figsize=(12, 2 * rows))
    plot_idx = 1
    for label_idx in range(rows * len(class_names)):
        example_idx = examples[label_idx % len(class_names)][label_idx // len(class_names)]
        plt.subplot(rows, len(class_names), plot_idx)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(train.imgs[example_idx]), cmap='gray')
        plt.xlabel(class_names[label_idx % len(class_names)])
        plot_idx += 1

    plt.savefig(os.path.join(save_dir, 'dataset_image_examples.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 4))
    sets = [train, val, test]
    set_names = ["Train", "Validation", "Test"]
    for i, dataset in enumerate(sets):
        class_counts = np.zeros(len(class_names), dtype=int)
        for label in dataset.labels:
            class_counts[int(label[0])] += 1
        plt.subplot(1, 3, i + 1)
        plt.bar(class_names, class_counts, color=['blue', 'orange', 'green', 'red'])
        plt.title(f"{set_names[i]} Set Class Distribution")

    plt.savefig(os.path.join(save_dir, 'dataset_class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_curves(history: dict, save_path: str):
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    ax.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(model, data_loader, class_names, device, save_path: str, normalize=True):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    cm = confusion_matrix(all_true, all_preds, normalize='true' if normalize else None)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    ax.set_title(title)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_comparison(results: list, save_path: str, top_n: int = 20):
    if not results:
        return

    top_results = results[:top_n]

    fig, ax = plt.subplots(figsize=(14, 8))

    names = [r["name"] for r in top_results]
    val_accs = [r["best_val_acc"] for r in top_results]
    train_accs = [r["final_train_acc"] for r in top_results]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, val_accs, width, label="Val Acc", color="steelblue")
    ax.bar(x + width/2, train_accs, width, label="Train Acc", color="coral")

    ax.set_ylabel("Accuracy")
    ax.set_title(f"Top {len(top_results)} Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
