config = {
	# Model
	"in_channels": 3,
	"num_classes": 4,

	# Data
	"image_size": 64,
	"subsample_train": None,  # int or None

	# Training
	"batch_size": 64,
	"epochs": 50,
	"lr": 1e-3,
	"seed": 0,

	# Device: 'cpu' or 'cuda'. Set to 'cuda' if you want GPU and it's available.
	"device": "cpu",
}
