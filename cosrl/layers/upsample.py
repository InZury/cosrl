import torch.nn as nn
import torch.nn.functional as func


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        scale_factor=4,
        mode="bilinear",
        align_corners=False
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, size=None):
        if size is not None:
            return func.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        else:
            return func.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)