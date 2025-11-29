import os
from medmnist import OCTMNIST
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
from torchvision import transforms

def load_data(normalize=True, augment_minority=True):
    train = OCTMNIST("train", download = True, size=64)
    val = OCTMNIST("val", download = True, size=64)
    test = OCTMNIST("test", download = True, size=64)

    visualize_data(train, val, test)

    x_train = np.array(train.imgs).astype(np.float32)
    x_train = x_train[:, None, :, :]   # shape → (N, 1, 64, 64)
    y_train = np.array(train.labels).ravel().astype(np.int64)

    x_val = np.array(val.imgs).astype(np.float32)
    x_val = x_val[:, None, :, :]       # shape → (N, 1, 64, 64)
    y_val = np.array(val.labels).ravel().astype(np.int64)

    x_test = np.array(test.imgs).astype(np.float32)
    x_test = x_test[:, None, :, :]     # shape → (N, 1, 64, 64)
    y_test = np.array(test.labels).ravel().astype(np.int64)

    # Data augmentation for minority classes
    if augment_minority:
        counter = Counter(y_train)
        max_count = max(counter.values())

        aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])

        augmented_images = []
        augmented_labels = []

        for cls, count in counter.items():
            if count < max_count:
                # Find indices of this class
                indices = np.where(y_train == cls)[0]
                # Number of new samples to generate
                n_aug = max_count - count

                for i in range(n_aug):
                    idx = indices[i % len(indices)]
                    img = x_train[idx, 0, :, :]  # remove channel dim for PIL
                    img_aug = aug_transform(img.astype(np.uint8))  # PIL expects uint8
                    img_aug = np.array(img_aug.numpy()[0], dtype=np.float32)  # back to numpy
                    augmented_images.append(img_aug[None, :, :])  # add channel dim
                    augmented_labels.append(cls)

        if augmented_images:
            x_train = np.concatenate([x_train, np.array(augmented_images)], axis=0)
            y_train = np.concatenate([y_train, np.array(augmented_labels)], axis=0)
            print(f"After augmentation: train size={len(x_train)}")

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

    # Collect 2 examples of each label type
    examples = {0: [], 1: [], 2: [], 3: []}
    for i, label in enumerate(train.labels):
        label_idx = int(label[0])
        if len(examples[label_idx]) < rows:
            examples[label_idx].append(i)
        if all(len(examples[j]) == rows for j in range(4)):
            break

    # Plot 2 examples of each label
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

#load_data()