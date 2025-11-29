import os
import json
import torch
from datetime import datetime


class Experiment:

    def __init__(self, name: str, base_dir: str = "experiments"):
        """
        Create a new experiment.

        Args:
            name: Name (e.g. "baseline_normalized")
            base_dir: Root directory for all experiments
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.name = f"{timestamp}_{name}"
        self.dir = os.path.join(base_dir, self.name)

        # Create the folder structure
        self.weights_dir = os.path.join(self.dir, "weights")
        self.plots_dir = os.path.join(self.dir, "plots")

        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Init the log file
        self.log_path = os.path.join(self.dir, "log.txt")

        print(f"[Experiment] Created: {self.dir}")

    def save_config(self, config: dict):
        """Save configuration to JSON file."""
        config_path = os.path.join(self.dir, "config.json")

        serializable_config = {}
        for key, value in config.items():
            try:
                json.dumps(value)
                serializable_config[key] = value
            except (TypeError, ValueError):
                serializable_config[key] = str(value)

        with open(config_path, "w") as f:
            json.dump(serializable_config, f, indent=2)

        print(f"[Experiment] Saved config: {config_path}")

    def log(self, message: str, also_print: bool = True):
        """Append message to log file."""
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

        if also_print:
            print(message)

    def save_checkpoint(self, model, optimizer, epoch: int, val_acc: float,
                        val_loss: float = None, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            model: Model
            optimizer: Optimizer
            epoch: Current epoch
            val_acc: Validation accuracy
            val_loss: Validation loss (optional)
            is_best: If True, save as 'best.pth', otherwise 'final.pth'
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
        }

        filename = "weights_best.pth" if is_best else "weights_final.pth"
        path = os.path.join(self.weights_dir, filename)
        torch.save(checkpoint, path)

        if is_best:
            self.log(f"[Checkpoint] New best model saved (val_acc={val_acc:.4f})")

    def save_plot(self, fig, filename: str):
        """
        Save matplotlib figure to plots folder.

        Args:
            fig: Matplotlib figure
            filename: Name of file (such as "accuracy.png")
        """
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Experiment] Saved plot: {path}")

    def get_weights_path(self, which: str = "best") -> str:
        """Get path to saved weights."""
        return os.path.join(self.weights_dir, f"{which}.pth")

    def get_plots_dir(self) -> str:
        """Get path to plots directory."""
        return self.plots_dir


def load_experiment(experiment_dir: str) -> dict:
    """
    Load config and weights from a previous experiment.

    Args:
        experiment_dir: Path to experiment folder

    Returns:
        Dict with 'config' and 'checkpoint' keys
    """
    config_path = os.path.join(experiment_dir, "config.json")
    weights_path = os.path.join(experiment_dir, "weights", "best.pth")

    with open(config_path, "r") as f:
        config = json.load(f)

    checkpoint = torch.load(weights_path, map_location="cpu")

    print(f"[Experiment] Loaded: {experiment_dir}")
    print(f"  Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f}")

    return {
        "config": config,
        "checkpoint": checkpoint,
    }


def list_experiments(base_dir: str = "experiments") -> list:
    if not os.path.exists(base_dir):
        return []

    experiments = []
    for name in sorted(os.listdir(base_dir)):
        exp_dir = os.path.join(base_dir, name)
        if os.path.isdir(exp_dir):
            config_path = os.path.join(exp_dir, "config.json")
            weights_path = os.path.join(exp_dir, "weights", "best.pth")

            info = {"name": name, "dir": exp_dir}

            # Try to load val_acc from checkpoint
            if os.path.exists(weights_path):
                try:
                    checkpoint = torch.load(weights_path, map_location="cpu")
                    info["val_acc"] = checkpoint.get("val_acc", None)
                    info["epoch"] = checkpoint.get("epoch", None)
                except:
                    pass

            experiments.append(info)

    return experiments
