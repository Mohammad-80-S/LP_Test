import torch
import torch.nn as nn


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn_layer = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        return self.act_layer(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = BasicConvBlock(channels, channels)
        self.sec_layer = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn_layer = nn.BatchNorm2d(channels, momentum=0.99, eps=0.001)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv_block(x)
        h = self.sec_layer(h)
        h = self.bn_layer(h)
        return self.act_layer(x + h)


class WPODNet(nn.Module):
    """WPODNet model for license plate detection."""
    
    stride = 16
    scale_factor = 7.75

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            BasicConvBlock(3, 16),
            BasicConvBlock(16, 16),
            nn.MaxPool2d(2),
            BasicConvBlock(16, 32),
            ResBlock(32),
            nn.MaxPool2d(2),
            BasicConvBlock(32, 64),
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2),
            BasicConvBlock(64, 64),
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2),
            BasicConvBlock(64, 128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
        )
        self.prob_layer = nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.bbox_layer = nn.Conv2d(128, 6, kernel_size=3, padding=1)
        self.register_buffer("dummy", torch.Tensor(), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def forward(self, image: torch.Tensor):
        feature = self.backbone(image)
        probs = self.prob_layer(feature)
        probs = torch.softmax(probs, dim=1)
        affines = self.bbox_layer(feature)
        return probs, affines


def load_wpodnet_from_checkpoint(ckpt_path: str) -> WPODNet:
    """Load WPODNet from checkpoint."""
    model = WPODNet()
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint)
    return model