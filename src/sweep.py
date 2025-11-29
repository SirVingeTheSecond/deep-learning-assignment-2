"""
Grid Search for CNN.

Usage:
    python sweep.py                    # Screen all, then full train passed
    python sweep.py --continue-on-pass # Continue to full epochs immediately if passed
    python sweep.py --list             # Show all combinations
    python sweep.py --count            # Count combinations
"""

import os
import json
import itertools
import argparse
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from config import config as base_config
from data import load_data


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


sweep_config = base_config.get("sweep", {})
PARAM_GRID = sweep_config.get("param_grid", {
    "use_batchnorm": [False, True],
    "dropout": [0.0, 0.3, 0.5],
    "num_blocks": [2, 3, 4],
    "lr": [1e-4, 1e-3, 1e-2],
    "weight_decay": [0, 1e-4, 5e-4],
})
SCREENING_EPOCHS = sweep_config.get("screening_epochs", 10)
SCREENING_THRESHOLD = sweep_config.get("screening_threshold", 0.90)
FULL_EPOCHS = sweep_config.get("full_epochs", 50)

early_stop_config = base_config.get("early_stopping", {})
EARLY_STOPPING_ENABLED = early_stop_config.get("enabled", True)
EARLY_STOPPING_PATIENCE = early_stop_config.get("patience", 5)
EARLY_STOPPING_MIN_DELTA = early_stop_config.get("min_delta", 0.001)


class ConfigurableCNN(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        num_blocks: int = 2,
        use_batchnorm: bool = False,
        dropout: float = 0.5,
        base_filters: int = 32,
    ):
        super().__init__()

        layers = []
        in_ch = in_channels
        out_ch = base_filters

        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)

        self.features = nn.Sequential(*layers)

        final_size = 64 // (2 ** num_blocks)
        flatten_dim = in_ch * final_size * final_size

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            classifier_layers.append(nn.Dropout(p=dropout))
        classifier_layers.append(nn.Linear(128, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)
        self._init_weights()

    def forward(self, x):
        return self.classifier(self.features(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_all_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        config["name"] = _config_to_name(config)
        combinations.append(config)

    return combinations


def _config_to_name(config: Dict[str, Any]) -> str:
    parts = []
    if config.get("use_batchnorm"):
        parts.append("bn")
    parts.append(f"d{config['dropout']}")
    parts.append(f"b{config['num_blocks']}")
    parts.append(f"lr{config['lr']}")
    parts.append(f"wd{config['weight_decay']}")
    return "_".join(parts)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (out.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

    return running_loss / total, correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

    return running_loss / total, correct / total


def run_experiment_with_screening(
    exp_config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    save_dir: str,
    screening_epochs: int,
    full_epochs: int,
    threshold: float,
    continue_on_pass: bool,
) -> Dict[str, Any]:
    name = exp_config["name"]
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    model = ConfigurableCNN(
        in_channels=base_config.get("in_channels", 1),
        num_classes=base_config.get("num_classes", 4),
        num_blocks=exp_config["num_blocks"],
        use_batchnorm=exp_config["use_batchnorm"],
        dropout=exp_config["dropout"],
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=exp_config["lr"],
        weight_decay=exp_config["weight_decay"],
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA) if EARLY_STOPPING_ENABLED else None
    stopped_early = False

    # Screening phase
    for epoch in range(1, screening_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)

        print(f"Epoch {epoch:3d}/{screening_epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    passed = best_val_acc >= threshold
    status = "PASSED" if passed else "FAILED"
    print(f"Screening: {status} ({best_val_acc:.4f} vs {threshold:.4f})")

    # Continue training if passed and continue_on_pass is True
    if passed and continue_on_pass and full_epochs > screening_epochs:
        print(f"\nContinuing to {full_epochs} epochs...")
        for epoch in range(screening_epochs + 1, full_epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            best_val_acc = max(best_val_acc, val_acc)

            print(f"Epoch {epoch:3d}/{full_epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

            if early_stopping and early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch}")
                stopped_early = True
                break

    results = {
        "name": name,
        "config": exp_config,
        "epochs": len(history["train_loss"]),
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
        "passed_screening": passed,
        "stopped_early": stopped_early,
        "history": history,
        "parameters": count_parameters(model),
    }

    exp_dir = os.path.join(save_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(exp_dir, "weights.pt"))

    return results


def run_full_training(
    exp_config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    save_dir: str,
    epochs: int,
) -> Dict[str, Any]:
    name = exp_config["name"]
    print(f"\n{'='*60}")
    print(f"Experiment: {name} (full training)")
    print(f"{'='*60}")

    model = ConfigurableCNN(
        in_channels=base_config.get("in_channels", 1),
        num_classes=base_config.get("num_classes", 4),
        num_blocks=exp_config["num_blocks"],
        use_batchnorm=exp_config["use_batchnorm"],
        dropout=exp_config["dropout"],
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=exp_config["lr"],
        weight_decay=exp_config["weight_decay"],
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA) if EARLY_STOPPING_ENABLED else None
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)

        print(f"Epoch {epoch:3d}/{epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        if early_stopping and early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            stopped_early = True
            break

    results = {
        "name": name,
        "config": exp_config,
        "epochs": len(history["train_loss"]),
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "final_train_acc": history["train_acc"][-1],
        "stopped_early": stopped_early,
        "history": history,
        "parameters": count_parameters(model),
    }

    exp_dir = os.path.join(save_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(exp_dir, "weights.pt"))

    return results


def run_grid_search(
    continue_on_pass: bool = False,
    screening_epochs: int = SCREENING_EPOCHS,
    screening_threshold: float = SCREENING_THRESHOLD,
    full_epochs: int = FULL_EPOCHS,
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = f"sweeps/{timestamp}"
    os.makedirs(sweep_dir, exist_ok=True)

    device = base_config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"

    print(f"Device: {device}")
    print(f"Sweep directory: {sweep_dir}")
    print(f"Mode: {'continue-on-pass' if continue_on_pass else 'screen-all-first'}")

    seed = base_config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("\nLoading data...")
    x_train, y_train, x_val, y_val, _, _ = load_data(normalize=True)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())

    batch_size = base_config.get("batch_size", 128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_configs = generate_all_combinations(PARAM_GRID)
    print(f"Total configurations: {len(all_configs)}")

    with open(os.path.join(sweep_dir, "sweep_config.json"), "w") as f:
        json.dump({
            "param_grid": {k: [str(v) for v in vals] for k, vals in PARAM_GRID.items()},
            "screening_epochs": screening_epochs,
            "screening_threshold": screening_threshold,
            "full_epochs": full_epochs,
            "continue_on_pass": continue_on_pass,
            "total_combinations": len(all_configs),
        }, f, indent=2)

    all_results = []

    if continue_on_pass:
        # Screen each config and immediately continue if passed
        for i, exp_config in enumerate(all_configs):
            print(f"\n[{i+1}/{len(all_configs)}]")
            results = run_experiment_with_screening(
                exp_config, train_loader, val_loader, device, sweep_dir,
                screening_epochs, full_epochs, screening_threshold, continue_on_pass=True
            )
            all_results.append(results)
    else:
        # Screen all first
        print(f"\n{'#'*60}")
        print(f"PHASE 1: SCREENING ({screening_epochs} epochs, threshold={screening_threshold:.0%})")
        print(f"{'#'*60}")

        for i, exp_config in enumerate(all_configs):
            print(f"\n[{i+1}/{len(all_configs)}]")
            results = run_experiment_with_screening(
                exp_config, train_loader, val_loader, device,
                os.path.join(sweep_dir, "screening"),
                screening_epochs, full_epochs, screening_threshold, continue_on_pass=False
            )
            all_results.append(results)

        passed_configs = [r["config"] for r in all_results if r["passed_screening"]]
        print(f"\nScreening complete: {len(passed_configs)}/{len(all_configs)} passed")

        if passed_configs:
            print(f"\n{'#'*60}")
            print(f"PHASE 2: FULL TRAINING ({full_epochs} epochs)")
            print(f"{'#'*60}")

            full_results = []
            for i, exp_config in enumerate(passed_configs):
                print(f"\n[{i+1}/{len(passed_configs)}]")
                results = run_full_training(
                    exp_config, train_loader, val_loader, device,
                    os.path.join(sweep_dir, "full"), full_epochs
                )
                full_results.append(results)

            all_results = full_results

    _save_summary(all_results, sweep_dir)
    _generate_comparison_plot(all_results, sweep_dir)
    _print_leaderboard(all_results)

    return all_results


def _save_summary(results: List[Dict], save_dir: str):
    sorted_results = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
    summary = {
        "total": len(results),
        "results": [
            {"name": r["name"], "best_val_acc": r["best_val_acc"], "epochs": r["epochs"], "params": r["parameters"]}
            for r in sorted_results
        ],
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _generate_comparison_plot(results: List[Dict], save_dir: str):
    if not results:
        return

    sorted_results = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(14, 8))

    names = [r["name"] for r in sorted_results]
    val_accs = [r["best_val_acc"] for r in sorted_results]
    train_accs = [r["final_train_acc"] for r in sorted_results]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, val_accs, width, label="Val Acc", color="steelblue")
    ax.bar(x + width/2, train_accs, width, label="Train Acc", color="coral")

    ax.set_ylabel("Accuracy")
    ax.set_title(f"Top {len(sorted_results)} Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
    plt.close(fig)


def _print_leaderboard(results: List[Dict], top_n: int = 10):
    sorted_results = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)

    print(f"\n{'='*80}")
    print(f"TOP {top_n} RESULTS")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Name':<40}{'Val Acc':>12}{'Epochs':>8}{'Params':>12}")
    print(f"{'-'*80}")

    for i, r in enumerate(sorted_results[:top_n], 1):
        print(f"{i:<6}{r['name']:<40}{r['best_val_acc']:>12.4f}{r['epochs']:>8}{r['parameters']:>12,}")


def list_all_combinations():
    configs = generate_all_combinations(PARAM_GRID)
    print(f"\nTotal combinations: {len(configs)}\n")
    print(f"{'Name':<45} {'BN':<6} {'Drop':<6} {'Blocks':<8} {'LR':<10} {'WD'}")
    print("-" * 85)
    for c in configs:
        print(f"{c['name']:<45} {str(c['use_batchnorm']):<6} {c['dropout']:<6} "
              f"{c['num_blocks']:<8} {c['lr']:<10} {c['weight_decay']}")


def count_combinations():
    total = 1
    for key, values in PARAM_GRID.items():
        print(f"  {key}: {len(values)} values {values}")
        total *= len(values)
    print(f"\nTotal: {total} combinations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search sweep")

    parser.add_argument("--continue-on-pass", action="store_true",
                        help="Continue to full epochs immediately when screening passes")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--screening-epochs", type=int, default=SCREENING_EPOCHS)
    parser.add_argument("--threshold", type=float, default=SCREENING_THRESHOLD)
    parser.add_argument("--full-epochs", type=int, default=FULL_EPOCHS)

    args = parser.parse_args()

    if args.list:
        list_all_combinations()
    elif args.count:
        count_combinations()
    else:
        run_grid_search(
            continue_on_pass=args.continue_on_pass,
            screening_epochs=args.screening_epochs,
            screening_threshold=args.threshold,
            full_epochs=args.full_epochs,
        )
