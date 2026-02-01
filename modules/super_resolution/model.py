import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = nn.Parameter(torch.FloatTensor([0.2]))

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        return x3 * self.res_scale + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels, num_convs=5):
        super().__init__()
        self.rdb_layers = nn.Sequential(
            *[ResidualDenseBlock(in_channels, growth_channels) for _ in range(num_convs)]
        )
        self.res_scale = nn.Parameter(torch.FloatTensor([0.2]))

    def forward(self, x):
        out = self.rdb_layers(x)
        return out * self.res_scale + x


# class ECALayer(nn.Module):
#     """Channel Attention (SE-style)."""
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         hidden = max(1, channels // reduction)
#         self.fc1 = nn.Linear(channels, hidden, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(hidden, channels, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc2(self.relu(self.fc1(y)))
#         y = self.sigmoid(y).view(b, c, 1, 1)
#         return x * y

# ECALayer
class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.k_size = k_size

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def get_k_size(self):
        return self.k_size


class SuperResolutionModel(nn.Module):
    """Super Resolution model with residual dense blocks."""
    
    def __init__(self, num_blocks, in_channels, growth_channels, scale_factor):
        super().__init__()
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        
        self.initial_conv = nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1)
        self.rdbs = nn.Sequential(
            *[ResidualInResidualDenseBlock(growth_channels, growth_channels, num_convs=5) 
              for _ in range(num_blocks)]
        )
        self.channel_attention = ECALayer(growth_channels)
        self.scale_factor = scale_factor
        
        if scale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(growth_channels, growth_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        elif scale_factor == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(growth_channels, growth_channels * 16, kernel_size=3, padding=1),
                nn.PixelShuffle(4)
            )
        elif scale_factor == 8:
            self.upsample = nn.Sequential(
                nn.Conv2d(growth_channels, growth_channels * 64, kernel_size=3, padding=1),
                nn.PixelShuffle(8)
            )
        else:
            raise ValueError("Scale factor must be 2, 4, or 8.")
        
        self.final_conv = nn.Conv2d(growth_channels, in_channels, kernel_size=3, padding=1)
        self.num_blocks = num_blocks
        self.growth_channels = growth_channels
        self.in_channels = in_channels

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.initial_conv(x)
        x = self.rdbs(x)
        x = self.channel_attention(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x