import torch
import torch.nn as nn
import torch.nn.functional as func

from cosrl.layers import PointWiseLayer, DSConvLayer, WindowAttentionBlock, UpsampleLayer


class Stage(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        input_resolution,
        num_heads,
        num_blocks=2,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attention_drop=0.0,
        projection_drop=0.0,
        drop_path=0.0,
        activation=nn.GELU,
        norm=nn.LayerNorm
    ):
        super().__init__()
        self.fusion_layer = nn.Conv2d(
            in_channels=in_channels+skip_channels, out_channels=out_channels, kernel_size=1, bias=False
        )
        self.pre_refine_layer = DSConvLayer(in_channels=out_channels, out_channels=out_channels)

        self.blocks = nn.ModuleList([
            WindowAttentionBlock(
                channels=out_channels,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (index % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attention_drop=attention_drop,
                projection_drop=projection_drop,
                drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                activation=activation,
                norm=norm,
            ) for index in range(num_blocks)
        ])

        self.post_refine_layer = DSConvLayer(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, skip):
        x = func.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion_layer(x)
        x = self.pre_refine_layer(x)

        for block in self.blocks:
            x = block(x)

        x = self.post_refine_layer(x)

        return x


class SwinDecoder(nn.Module):
    def __init__(
        self,
        in_channels=(768, 384, 192, 96),
        hidden_channels=256,
        input_resolution=352,
        num_stages=3,
        num_heads=(8, 8, 16),
        num_attention=(2, 2, 1),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attention_drop=0.0,
        projection_drop=0.0,
        drop_path=0.0,
        activation=nn.GELU,
        norm=nn.LayerNorm,
    ):
        super().__init__()
        self.pointwise_layers = nn.ModuleList([
            PointWiseLayer(in_channels=in_channel, out_channels=hidden_channels)
            for in_channel in in_channels
        ])

        self.dsconv_layer = DSConvLayer(in_channels=hidden_channels, out_channels=hidden_channels)

        self.stages = nn.ModuleList([
            Stage(
                in_channels=hidden_channels,
                skip_channels=hidden_channels,
                out_channels=hidden_channels,
                input_resolution=input_resolution,
                num_heads=num_heads[index],
                num_blocks=num_attention[index],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attention_drop=attention_drop,
                projection_drop=projection_drop,
                drop_path=drop_path,
                activation=activation,
                norm=norm
            ) for index in range(num_stages)
        ])

        self.mask_head = nn.Sequential(
            DSConvLayer(in_channels=hidden_channels, out_channels=hidden_channels),
            nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1),
            UpsampleLayer(scale_factor=4, mode="bilinear", align_corners=False)
        )
        self.boundary_head = nn.Sequential(
            DSConvLayer(in_channels=hidden_channels, out_channels=hidden_channels),
            nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1),
            UpsampleLayer(scale_factor=4, mode="bilinear", align_corners=False)
        )

    def forward(self, features):
        features = [self.pointwise_layers[index](features[index]) for index in range(len(features))]

        x = self.bottom_layer(features[0])

        for index in range(len(self.stages)):
            x = self.stages[index](x, skip=features[index+1])

        mask_logit = self.mask_head(x)
        boundary_logit = self.boundary_head(x)

        return mask_logit, boundary_logit
