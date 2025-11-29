import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import config
from data import load_data
from cnn import CNN, count_parameters
from experiment import Experiment
from training import create_dataloaders, train_one_epoch, validate, get_early_stopping


def plot_training_curves(history: dict, experiment: Experiment):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    experiment.save_plot(fig, "loss.png")
    plt.close(fig)

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    ax.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    experiment.save_plot(fig, "accuracy.png")
    plt.close(fig)


def plot_confusion_matrix(model, val_loader, device, experiment: Experiment):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    cm = confusion_matrix(all_true, all_preds, normalize='true')

    class_names = config.get("class_names", ["CNV", "DME", "Drusen", "Normal"])

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Normalized)")
    experiment.save_plot(fig, "confusion_matrix.png")
    plt.close(fig)


def train(experiment_name: str, normalize: bool = True):
    # Create experiment (this is auto-generated)
    exp = Experiment(experiment_name)

    # Build run config which is a copy of base config + runtime settings
    run_config = config.copy()
    run_config["normalize"] = normalize
    run_config["experiment_name"] = experiment_name
    exp.save_config(run_config)

    # Log the start of the experiment start
    exp.log("=" * 30)
    exp.log(f"Experiment: {experiment_name}")
    exp.log(f"Normalize: {normalize}")
    exp.log("=" * 30)

    # Seed
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Device
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        exp.log("Since CUDA is not available, I will be using your slow ass CPU ⎛⎝( ` ᢍ ´ )⎠⎞ᵘʰᵃʰᵃ")
    exp.log(f"Device: {device}")

    # Load data
    exp.log("\nLoading data...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(normalize=normalize)
    exp.log(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # Create dataloaders
    batch_size = config.get("batch_size", 128)
    train_loader, val_loader = create_dataloaders(
        x_train, y_train, x_val, y_val, batch_size
    )

    # Init model
    in_channels = config.get("in_channels", 1)
    num_classes = config.get("num_classes", 4)

    model = CNN(in_channels=in_channels, num_classes=num_classes)
    model.to(device)

    # Init lazy layers
    image_size = config.get("image_size", 64)
    dummy = torch.randn(1, in_channels, image_size, image_size).to(device)
    model(dummy)

    # Init weights
    model._init_weights()

    exp.log(f"\nModel parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 0)
    )

    # Early stopping
    early_stopping = get_early_stopping()
    if early_stopping:
        es_config = config.get("early_stopping", {})
        exp.log(f"Early stopping: patience={es_config.get('patience')}, min_delta={es_config.get('min_delta')}")

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    epochs = config.get("epochs", 50)

    exp.log(f"\nTraining for {epochs} epochs...")
    exp.log("-" * 60)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Append to the history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Log the progress, although it might not be too relevant?
        exp.log(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            exp.save_checkpoint(
                model, optimizer, epoch, val_acc, val_loss, is_best=True
            )

        # Early stopping check
        if early_stopping and early_stopping(val_loss):
            exp.log(f"Early stopping triggered at epoch {epoch}")
            break

    # Save the final model
    exp.save_checkpoint(
        model, optimizer, epoch, val_acc, val_loss, is_best=False
    )

    # Generate plots
    exp.log("\nGenerating plots...")
    plot_training_curves(history, exp)
    plot_confusion_matrix(model, val_loader, device, exp)

    # Summary
    exp.log("\n" + "=" * 30)
    exp.log("Training Complete")
    exp.log(f"Best validation accuracy: {best_val_acc:.4f}")
    exp.log(f"Experiment saved to: {exp.dir}")
    exp.log("=" * 30)

    return exp, model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the damn CNN")
    parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable input normalization"
    )

    args = parser.parse_args()

    train(
        experiment_name=args.name,
        normalize=not args.no_normalize
    )
