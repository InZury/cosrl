import torch.nn as nn


class PointWiseLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False
    ):
        super().__init__()
        self.layer = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):
        return self.layer(x)

    def flops(self, height, width):
        in_channels = self.layer.in_channels
        out_channels = self.layer.out_channels

        return int(height * width * in_channels * out_channels)
