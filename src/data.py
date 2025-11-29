import os
import numpy as np
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader
from medmnist import OCTMNIST
from matplotlib import pyplot as plt


def load_data(normalize=True):
    train = OCTMNIST("train", download=True, size=64)
    val = OCTMNIST("val", download=True, size=64)
    test = OCTMNIST("test", download=True, size=64)

    visualize_data(train, val, test)

    x_train = np.array(train.imgs).astype(np.float32)
    x_train = x_train[:, None, :, :]  # shape -> (N, 1, 64, 64)
    y_train = np.array(train.labels).ravel().astype(np.int64)

    x_val = np.array(val.imgs).astype(np.float32)
    x_val = x_val[:, None, :, :]  # shape -> (N, 1, 64, 64)
    y_val = np.array(val.labels).ravel().astype(np.int64)

    x_test = np.array(test.imgs).astype(np.float32)
    x_test = x_test[:, None, :, :]  # shape -> (N, 1, 64, 64)
    y_test = np.array(test.labels).ravel().astype(np.int64)

    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    print(f"Class distribution: {dict(sorted(Counter(y_train).items()))}")

    # Normalization using the training set only
    if normalize:
        mean = x_train.mean()
        std = x_train.std()
        print(f"[Before] mean={mean:.2f}, std={std:.2f}")

        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std

        print(f"[After] train mean={x_train.mean():.4f}, std={x_train.std():.4f}")

    return x_train, y_train, x_val, y_val, x_test, y_test


def visualize_data(train, val, test):
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    labels = ["CNV", "DME", "Drusen", "Normal"]
    rows = 4

    # Collect examples of each label type
    examples = {0: [], 1: [], 2: [], 3: []}
    for i, label in enumerate(train.labels):
        label_idx = int(label[0])
        if len(examples[label_idx]) < rows:
            examples[label_idx].append(i)
        if all(len(examples[j]) == rows for j in range(4)):
            break

    # Plot examples of each label
    plt.figure(figsize=(12, 2 * rows))
    plot_idx = 1
    for label_idx in range(rows * 4):
        example_idx = examples[label_idx % 4][label_idx // 4]
        plt.subplot(rows, 4, plot_idx)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(train.imgs[example_idx]), cmap='gray')
        plt.xlabel(labels[label_idx % 4])
        plot_idx += 1

    plt.savefig(os.path.join(plots_dir, 'dataset_image_examples.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot class distribution in each set
    plt.figure(figsize=(12, 4))
    sets = [train, val, test]
    set_names = ["Train", "Validation", "Test"]
    for i, dataset in enumerate(sets):
        class_counts = np.zeros(4, dtype=int)
        for label in dataset.labels:
            class_counts[int(label[0])] += 1
        plt.subplot(1, 3, i + 1)
        plt.bar(labels, class_counts, color=['blue', 'orange', 'green', 'red'])
        plt.title(f"{set_names[i]} Set Class Distribution")

    plt.savefig(os.path.join(plots_dir, 'dataset_class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_dataloaders(x_train, y_train, x_val, y_val, batch_size: int):
    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val).long()
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
