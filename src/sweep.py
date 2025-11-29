"""
Grid Search for CNN.

Usage:
    Single machine:
    python sweep.py --sweep-name test --continue-on-pass

    Distributed across machines:
    python sweep.py --sweep-name test --partition x/3 --continue-on-pass

    (REMEMBER TO SYNC RESULTS FIRST) Merge the results from all machines:
    python sweep.py --merge v1

    Other:
    python sweep.py --list
    python sweep.py --count
"""

import os
import json
import itertools
import argparse
from datetime import datetime
from typing import Dict, Any, List
import platform

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from config import config as base_config
from data import load_data
from models import ConfigurableCNN
from training import train_one_epoch, evaluate, EarlyStopping
from visualization import plot_sweep_comparison
from utils import count_parameters


sweep_config = base_config["sweep"]
PARAM_GRID = sweep_config["param_grid"]
SCREENING_EPOCHS = sweep_config["screening_epochs"]
SCREENING_THRESHOLD = sweep_config["screening_threshold"]
FULL_EPOCHS = sweep_config["full_epochs"]

early_stop_config = base_config["early_stopping"]
EARLY_STOPPING_ENABLED = early_stop_config["enabled"]
EARLY_STOPPING_PATIENCE = early_stop_config["patience"]
EARLY_STOPPING_MIN_DELTA = early_stop_config["min_delta"]


def generate_all_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    # Sort keys for deterministic order across machines
    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]

    combinations = []
    for i, combo in enumerate(itertools.product(*values)):
        config = dict(zip(keys, combo))
        config["name"] = _config_to_name(config)
        config["id"] = i
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


def get_results_path(sweep_dir: str, config_name: str) -> str:
    return os.path.join(sweep_dir, "results", config_name, "results.json")


def config_already_done(sweep_dir: str, config_name: str) -> bool:
    return os.path.exists(get_results_path(sweep_dir, config_name))


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
    print(f"Experiment: {name} (id={exp_config.get('id', '?')})")
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
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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
        "id": exp_config.get("id"),
        "config": exp_config,
        "history": history,
        "best_val_acc": best_val_acc,
        "final_train_acc": history["train_acc"][-1],
        "epochs": len(history["train_loss"]),
        "parameters": count_parameters(model),
        "passed_screening": passed,
        "stopped_early": stopped_early,
        "hostname": platform.node(),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = get_results_path(save_dir, name)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_grid_search(
    sweep_name: str,
    continue_on_pass: bool = False,
    screening_epochs: int = SCREENING_EPOCHS,
    screening_threshold: float = SCREENING_THRESHOLD,
    full_epochs: int = FULL_EPOCHS,
    partition: str = None,
):
    sweep_dir = f"sweeps/{sweep_name}"
    os.makedirs(sweep_dir, exist_ok=True)

    all_configs = generate_all_combinations(PARAM_GRID)

    # Partition configs for distributed runs
    if partition:
        partition_idx, total_partitions = map(int, partition.split('/'))
        partition_idx -= 1
        all_configs = all_configs[partition_idx::total_partitions]
        print(f"Running partition {partition_idx+1}/{total_partitions}: {len(all_configs)} configs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading data...")
    x_train, y_train, x_val, y_val, _, _ = load_data(normalize=True)

    batch_size = base_config.get("batch_size", 128)
    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val).long()
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Save sweep metadata
    with open(os.path.join(sweep_dir, "config.json"), "w") as f:
        json.dump({
            "sweep_name": sweep_name,
            "param_grid": PARAM_GRID,
            "screening_epochs": screening_epochs,
            "screening_threshold": screening_threshold,
            "full_epochs": full_epochs,
            "continue_on_pass": continue_on_pass,
            "partition": partition,
            "total_configs": len(all_configs),
            "device": device,
            "pytorch_version": torch.__version__,
        }, f, indent=2)

    all_results = []
    skipped = 0

    for i, exp_config in enumerate(all_configs):
        # Resume: skip if already done
        if config_already_done(sweep_dir, exp_config["name"]):
            print(f"[{i+1}/{len(all_configs)}] {exp_config['name']} - SKIPPED (already done)")
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(all_configs)}]")
        results = run_experiment_with_screening(
            exp_config, train_loader, val_loader, device, sweep_dir,
            screening_epochs, full_epochs, screening_threshold, continue_on_pass
        )
        all_results.append(results)

    print(f"\nCompleted: {len(all_results)}, Skipped: {skipped}")

    if all_results:
        _save_partition_summary(all_results, sweep_dir, partition)

    return all_results


def _save_partition_summary(results: List[Dict], sweep_dir: str, partition: str = None):
    sorted_results = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)

    suffix = f"_part{partition.replace('/', '-')}" if partition else ""
    summary_file = os.path.join(sweep_dir, f"summary{suffix}.json")

    summary = {
        "partition": partition,
        "hostname": platform.node(),
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "results": [
            {
                "name": r["name"],
                "id": r.get("id"),
                "best_val_acc": r["best_val_acc"],
                "epochs": r["epochs"],
                "params": r["parameters"],
                "passed_screening": r.get("passed_screening"),
                "stopped_early": r.get("stopped_early"),
            }
            for r in sorted_results
        ],
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to: {summary_file}")


def merge_results(sweep_name: str):
    sweep_dir = f"sweeps/{sweep_name}"
    results_dir = os.path.join(sweep_dir, "results")

    if not os.path.exists(results_dir):
        print(f"No results directory found at {results_dir}")
        return

    all_results = []

    for config_name in os.listdir(results_dir):
        results_file = os.path.join(results_dir, config_name, "results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                all_results.append(json.load(f))

    if not all_results:
        print("No results found to merge")
        return

    print(f"Found {len(all_results)} completed experiments")

    # Sort by val_acc
    sorted_results = sorted(all_results, key=lambda x: x["best_val_acc"], reverse=True)

    # Save merged summary
    summary = {
        "merged_at": datetime.now().isoformat(),
        "total": len(sorted_results),
        "results": [
            {
                "name": r["name"],
                "id": r.get("id"),
                "best_val_acc": r["best_val_acc"],
                "epochs": r["epochs"],
                "params": r["parameters"],
                "passed_screening": r.get("passed_screening"),
                "stopped_early": r.get("stopped_early"),
                "hostname": r.get("hostname"),
            }
            for r in sorted_results
        ],
    }

    with open(os.path.join(sweep_dir, "summary_merged.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Generate comparison plot
    plot_sweep_comparison(sorted_results, os.path.join(sweep_dir, "comparison.png"))

    # Print leaderboard
    _print_leaderboard(sorted_results)

    # Check for missing configs
    all_configs = generate_all_combinations(PARAM_GRID)
    completed_names = {r["name"] for r in all_results}
    missing = [c for c in all_configs if c["name"] not in completed_names]

    if missing:
        print(f"\nWARNING: {len(missing)} configs not yet completed:")
        for c in missing[:10]:
            print(f"  - {c['name']} (id={c['id']})")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


def _print_leaderboard(results: List[Dict], top_n: int = 10):
    print(f"\n{'='*85}")
    print(f"TOP {top_n} RESULTS")
    print(f"{'='*85}")
    print(f"{'Rank':<6}{'Name':<35}{'Val Acc':>10}{'Epochs':>8}{'Params':>10}{'Host':<15}")
    print(f"{'-'*85}")

    for i, r in enumerate(results[:top_n], 1):
        host = r.get("hostname", "?")[:14]
        print(f"{i:<6}{r['name']:<35}{r['best_val_acc']:>10.4f}{r['epochs']:>8}{r['parameters']:>10,}{host:<15}")


def list_all_combinations():
    configs = generate_all_combinations(PARAM_GRID)
    print(f"\nTotal combinations: {len(configs)}\n")
    print(f"{'ID':<5} {'Name':<40} {'BN':<6} {'Drop':<6} {'Blocks':<8} {'LR':<10} {'WD'}")
    print("-" * 95)
    for c in configs:
        print(f"{c['id']:<5} {c['name']:<40} {str(c['use_batchnorm']):<6} {c['dropout']:<6} "
              f"{c['num_blocks']:<8} {c['lr']:<10} {c['weight_decay']}")


def count_combinations():
    total = 1
    for key, values in sorted(PARAM_GRID.items()):
        print(f"  {key}: {len(values)} values {values}")
        total *= len(values)
    print(f"\nTotal: {total} combinations")
    print(f"\nWith 3 partitions: {total // 3} configs each (partition 3 gets {total - 2*(total//3)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search sweep")

    parser.add_argument("--sweep-name", type=str, default=None,
                        help="Name for this sweep (required for distributed runs)")
    parser.add_argument("--partition", type=str, default=None,
                        help="Run subset of configs, e.g. '1/3', '2/3', '3/3'")
    parser.add_argument("--continue-on-pass", action="store_true",
                        help="Continue to full epochs immediately when screening passes")
    parser.add_argument("--merge", type=str, default=None, metavar="SWEEP_NAME",
                        help="Merge results from all partitions for given sweep name")
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
    elif args.merge:
        merge_results(args.merge)
    else:
        if not args.sweep_name:
            # Generate default name if not distributed
            args.sweep_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        run_grid_search(
            sweep_name=args.sweep_name,
            continue_on_pass=args.continue_on_pass,
            screening_epochs=args.screening_epochs,
            screening_threshold=args.threshold,
            full_epochs=args.full_epochs,
            partition=args.partition,
        )
