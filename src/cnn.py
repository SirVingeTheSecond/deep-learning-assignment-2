
import torch
import torch.nn as nn


class CNN(nn.Module):
		

		def __init__(self, in_channels=3, num_classes=4, init_weights=True):
			super().__init__()
			self.features = nn.Sequential(
				nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2),
				nn.Conv2d(32, 64, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2),
			)

			self.classifier = nn.Sequential(
				nn.Flatten(),
				nn.Linear(64 * 16 * 16, 128),
				nn.ReLU(inplace=True),
				nn.Dropout(p=0.5),
				nn.Linear(128, num_classes),
			)

			if init_weights:
				self._init_weights()

		def forward(self, x):
			x = self.features(x)
			x = self.classifier(x)
			return x

		def _init_weights(self):
			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					if m.bias is not None:
						nn.init.zeros_(m.bias)
				elif isinstance(m, nn.Linear):
					nn.init.normal_(m.weight, 0.0, 0.01)
					if m.bias is not None:
						nn.init.zeros_(m.bias)


def count_parameters(model: nn.Module) -> int:
		return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
	# Quick sanity check when running the module directly.
	try:
		model = CNN(in_channels=3, num_classes=4)
		print(f"Model parameters: {count_parameters(model):,}")
		# Create a dummy batch with shape (batch, channels, height, width)
		x = torch.randn(2, 3, 64, 64)
		y = model(x)
		print("Forward output shape:", y.shape)
	except Exception as e:
		print("Error running SimpleCNN example:", e)
		raise
