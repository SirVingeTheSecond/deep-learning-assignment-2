import torch

config = {
    # Model
    "in_channels": 1,
    "num_classes": 4,
    "class_names": ["CNV", "DME", "Drusen", "Normal"],
    "dropout": 0.3,

    # Data
    "image_size": 64,

    # Training
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,

    "seed": 42,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

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
        "screening_threshold": 0.93,
        "full_epochs": 20,
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
