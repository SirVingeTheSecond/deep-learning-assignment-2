from medmnist import OCTMNIST
from matplotlib import pyplot as plt
import numpy as np

train = OCTMNIST("train", download = True, size=64)
val = OCTMNIST("val", download = True, size=64)
test = OCTMNIST("test", download = True, size=64)

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

plt.show()
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
    
plt.show()
plt.close()