import os
import numpy as np
from medmnist import OCTMNIST
import matplotlib.pyplot as plt
import random

def _to_numpy(ds):
    imgs = np.array(ds.imgs)
    y = np.array(ds.labels).ravel().astype(np.int64)
    return imgs, y


def _flatten(x):
    return x.reshape(x.shape[0], -1).astype(np.float32)


def _standardize(x_ref, x):
    mu = x_ref.mean(axis=0, keepdims=True)
    sd = x_ref.std(axis=0, keepdims=True) + 1e-8 # so we do not divide by zero
    return (x - mu) / sd

def plot_example_images(tr):
    plots_dir = "../plots"
    os.makedirs(plots_dir, exist_ok=True)

    trainImages,trainLabels,trainInfo = tr.__dict__['imgs'],tr.__dict__['labels'],tr.__dict__['info']['label']

    random.seed(42)
    fig, axes = plt.subplots(5, len(trainInfo), figsize=(15, 5))

    for class_,name in trainInfo.items():
        # Get indices of all images belonging to class i
        class_indices = [idx for idx, label in enumerate(trainLabels) if int(class_) == label]
        # Randomly select 5 indices
        selected_indices = random.sample(class_indices, 5)
        for j, idx in enumerate(selected_indices):
            image, label = trainImages[idx],trainLabels[idx]
            axes[j, int(class_)].imshow(image, cmap='gray')
            axes[j, int(class_)].axis('off')
            if j == 0:
                axes[j, int(class_)].set_title(f'{name[:5]}: {class_}')

    plt.tight_layout()
    out = f"{plots_dir}/example_images.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()


def load_data(size=28, subsample_train=None, seed=0, return_images=False):
    

    # DO THE SPLITS
    tr = OCTMNIST(split="train", download=True, size=size)
    va = OCTMNIST(split="val", download=True, size=size)
    te = OCTMNIST(split="test", download=True, size=size)

    plot_example_images(tr)

    Xtr_img, ytr = _to_numpy(tr)
    Xva_img, yva = _to_numpy(va)
    Xte_img, yte = _to_numpy(te)

    # If user wants image tensors for CNN: convert, standardize per-channel and return
    if return_images:
        # Convert to float32
        Xtr_f = Xtr_img.astype(np.float32)
        Xva_f = Xva_img.astype(np.float32)
        Xte_f = Xte_img.astype(np.float32)

       
        # If grayscale, add channel dimension
        if Xtr_f.ndim == 3:
            Xtr_f = Xtr_f[..., np.newaxis]
        if Xva_f.ndim == 3:
            Xva_f = Xva_f[..., np.newaxis]
        if Xte_f.ndim == 3:
            Xte_f = Xte_f[..., np.newaxis]

        if Xtr_f.ndim != 4 or Xva_f.ndim != 4 or Xte_f.ndim != 4:
            raise ValueError("Unexpected image array shape after channel handling")

        # Compute per-channel mean/std on training set. Expect images in HxWxC.
        mu = Xtr_f.mean(axis=(0, 1, 2), keepdims=True)
        sd = Xtr_f.std(axis=(0, 1, 2), keepdims=True) + 1e-8

        Xtr_f = (Xtr_f - mu) / sd
        Xva_f = (Xva_f - mu) / sd
        Xte_f = (Xte_f - mu) / sd

        # Reorder to (N, C, H, W)
        Xtr_f = np.transpose(Xtr_f, (0, 3, 1, 2)).astype(np.float32)
        Xva_f = np.transpose(Xva_f, (0, 3, 1, 2)).astype(np.float32)
        Xte_f = np.transpose(Xte_f, (0, 3, 1, 2)).astype(np.float32)

        # Optional subsample
        if subsample_train is not None:
            np.random.seed(seed)
            indices = np.random.choice(Xtr_f.shape[0], subsample_train, replace=False)
            Xtr_f = Xtr_f[indices]
            ytr = ytr[indices]

        return Xtr_f, ytr, Xva_f, yva, Xte_f, yte

    # Otherwise keep original flattened behaviour for compatibility
    # Flatten to vectors (28*28*3 = 2352)
    Xtr = _flatten(Xtr_img)
    Xva = _flatten(Xva_img)
    Xte = _flatten(Xte_img)

    # We can subsample training data
    if subsample_train is not None:
        np.random.seed(seed)
        indices = np.random.choice(Xtr.shape[0], subsample_train, replace=False)
        Xtr = Xtr[indices]
        ytr = ytr[indices]

    # We need to store the value of the original training statistics
    Xtr_original = Xtr.copy()

    # Standardize
    Xtr = _standardize(Xtr_original, Xtr)
    Xva = _standardize(Xtr_original, Xva) 
    Xte = _standardize(Xtr_original, Xte) 

    # Bias is added after standardization
    Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
    Xva = np.hstack([Xva, np.ones((Xva.shape[0], 1))])
    Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1))])

    return Xtr, ytr, Xva, yva, Xte, yte