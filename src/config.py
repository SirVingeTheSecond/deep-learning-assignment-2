import torch

config = {
	# Model
	"in_channels": 1,
	"num_classes": 4,

	# Data
	"image_size": 64,
	"subsample_train": None,  # Set to an int so we can debug quicker?

	# Training
    "batch_size": 128,
    "epochs": 20,
    "lr": 1e-3,

    # ToDo:
    # "weight_decay": 1e-4,
    # "dropout_rate": 0.5,

    "seed": 0,

	# Device:
	"device": "cuda" if torch.cuda.is_available() else "cpu",

    # Checkpoint
    "checkpoint_dir": "checkpoints",
    "save_best": True
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
