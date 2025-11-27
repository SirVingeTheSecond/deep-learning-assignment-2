import os
import numpy as np


# Import configurations
from config import (
    config,
)

from datav2 import load_data

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch import nn, optim
    from cnn import CNN, count_parameters
except Exception:
    # Defer import errors until runtime; main will print a helpful message.
    torch = None
# The filebuffer appends, remeber to add newlines \n for each output
logger = open("./output.log","a")
#logger.writelines("just works \n")

def train_quick():
    if torch is None:
        raise ImportError("PyTorch is required to run training. Install torch and torchvision.")

    # Seed
    np.random.seed(config.get("seed", 0))
    torch.manual_seed(config.get("seed", 0))

    # Load image tensors (N, C, H, W)
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    # Convert to torch
    Xtr_t = torch.from_numpy(x_train).to(torch.float32)
    ytr_t = torch.from_numpy(y_train).to(torch.long)
    Xva_t = torch.from_numpy(x_val).to(torch.float32)
    yva_t = torch.from_numpy(y_val).to(torch.long)

    print(f"Training data shape: {Xtr_t.shape}, Training labels shape: {ytr_t.shape}")


    train_ds = TensorDataset(Xtr_t, ytr_t)
    val_ds = TensorDataset(Xva_t, yva_t)

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model — infer number of channels from the training data
    inferred_in_channels = Xtr_t.shape[1]
    model = CNN(in_channels=inferred_in_channels,
                num_classes=config.get("num_classes", 4))
    model.to(device)

    image_size = config.get("image_size", 64)
    dummy = torch.randn(1, 1, image_size, image_size).to(device)
    model(dummy) # initializes Lazy layers

    model._init_weights()

    print(f"Model with {count_parameters(model):,} trainable parameters.")

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    criterion = nn.CrossEntropyLoss()

    # Train and record metrics for plotting
    epochs = config.get("epochs", 2)
    train_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train if total_train > 0 else 0.0

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()


                total += yb.size(0)

        val_acc = correct / total if total > 0 else 0.0

        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}/{epochs} - train_loss: {epoch_loss:.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

    torch.save(model.state_/dict(), "cnn_weights.pth")

    # Plot metrics using matplotlib (saved to plots/). Handle missing matplotlib gracefully.
    try:
        import matplotlib.pyplot as plt
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Train loss
        plt.figure()
        plt.plot(range(1, epochs + 1), train_losses, marker='o')
        plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        out_loss = os.path.join(plots_dir, 'train_loss.png')
        plt.savefig(out_loss, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved train loss plot: {out_loss}")

        # Accuracy: train vs val
        plt.figure()
        plt.plot(range(1, epochs + 1), train_accs, label='Train Acc', marker='o')
        plt.plot(range(1, epochs + 1), val_accs, label='Val Acc', marker='o')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        out_acc = os.path.join(plots_dir, 'accuracy.png')
        plt.savefig(out_acc, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved accuracy plot: {out_acc}")

        # generating prediction data on per class level
        all_preds = []
        all_true = []
        model.eval()

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                out = model(xb)
                preds = out.argmax(dim=1)

                # Move to CPU and convert to numpy to accumulate
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(yb.cpu().numpy())
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        # Compute the matrix
        cm = confusion_matrix(all_true, all_preds, normalize='true')

        # Define class names (If you know them, replace these strings)
        # Example: class_names = ["Healthy", "Mild", "Moderate", "Severe"]
        class_names = [str(i) for i in range(config.get("num_classes", 4))]

        # Setup the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)


        # Plot with a blue colormap
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

        plt.title('Confusion Matrix')

        # Save results
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        out_cm = os.path.join(plots_dir, 'confusion_matrix.png')

        plt.savefig(out_cm, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix: {out_cm}")

    except Exception as e:
        print("Skipped plotting (matplotlib missing or error):", e)


if __name__ == "__main__":
    # Convenience CLI: run a quick train when executing the module.
    try:
        train_quick()
    except Exception as e:
        print("Failed to run training:", e)
        raise
