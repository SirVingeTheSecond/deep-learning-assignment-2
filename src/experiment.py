import os
import json
from datetime import datetime
import torch


class Experiment:
    def __init__(self, name: str, base_dir: str = "experiments"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.name = f"{timestamp}_{name}"
        self.dir = os.path.join(base_dir, self.name)
        self.weights_dir = os.path.join(self.dir, "weights")
        self.plots_dir = os.path.join(self.dir, "plots")

        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.log_file = os.path.join(self.dir, "log.txt")
        self.config_file = os.path.join(self.dir, "config.json")

    def log(self, message: str):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def save_config(self, config: dict):
        serializable = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable[k] = v
            else:
                serializable[k] = str(v)

        with open(self.config_file, "w") as f:
            json.dump(serializable, f, indent=2)

    def save_checkpoint(self, model, optimizer, epoch, val_acc, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
        }

        filename = "best.pth" if is_best else "final.pth"
        path = os.path.join(self.weights_dir, filename)
        torch.save(checkpoint, path)

        if is_best:
            self.log(f"[Checkpoint] New best model saved (val_acc={val_acc:.4f})")

    def save_plot(self, fig, filename: str):
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self.log(f"[Plot] Saved: {path}")
