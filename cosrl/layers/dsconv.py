import torch.nn as nn

from cosrl.layers import PointWiseLayer

class DSConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=None,
        bias=False,
        norm=nn.BatchNorm2d,
        activation=nn.GELU
    ):
        super().__init__()
        groups = groups if groups is not None else in_channels

        self.depthwise = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias
        )
        self.pointwise = PointWiseLayer(in_channels=in_channels, out_channels=out_channels)
        self.norm = norm(out_channels) if norm is not None else nn.Identity()
        self.activation = activation() if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)

        return x
