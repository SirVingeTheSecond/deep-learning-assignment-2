import torch

config = {
    # Model
    "in_channels": 1,
    "num_classes": 4,
    "class_names": ["CNV", "DME", "Drusen", "Normal"],

    # Data
    "image_size": 64,
    "subsample_train": None,  # Set to an int so we can debug quicker?

    # Training
    "batch_size": 128,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,

    "seed": 0,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Checkpoint
    "checkpoint_dir": "checkpoints",
    "save_best": True,

    # Early stopping
    "early_stopping": {
        "enabled": True,
        "patience": 5,
        "min_delta": 0.001,
    },

    # Grid search
    "sweep": {
        "param_grid": {
            "use_batchnorm": [False, True],
            "dropout": [0.0, 0.3, 0.5],
            "num_blocks": [2, 3, 4],
            "lr": [1e-4, 1e-3, 1e-2],
            "weight_decay": [0, 1e-4, 5e-4],
        },
        "screening_epochs": 10,
        "screening_threshold": 0.90,
        "full_epochs": 50,
    },
}


def print_config():
    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
