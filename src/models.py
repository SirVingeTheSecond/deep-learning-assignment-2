import torch
import torch.nn as nn


def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CNN(nn.Module):

    def __init__(self, in_channels=1, num_classes=4, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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
        self.apply(_init_weights)

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    # Quick sanity check when running the module directly.
    from utils import count_parameters

    model = CNN(in_channels=1, num_classes=4)
    print(f"CNN parameters: {count_parameters(model):,}")
    # Create a dummy batch with shape (batch, channels, height, width)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print(f"CNN output shape: {y.shape}")

    model = ConfigurableCNN(in_channels=1, num_classes=4, num_blocks=3, use_batchnorm=True, dropout=0.5)
    print(f"ConfigurableCNN parameters: {count_parameters(model):,}")
    y = model(x)
    print(f"ConfigurableCNN output shape: {y.shape}")
