import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_

from cosrl.layers import DropPath


class WindowAttentionLayer(nn.Module):
    def __init__(
        self,
        channels,
        window_size,
        num_heads: int,
        qkv_bias=True,
        qk_scale=None,
        attention_drop=0.0,
        projection_drop=0.0
    ):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5

        self.position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        relative_coordinates = self.calculate_coordinates()
        self.register_buffer("relative_coordinates", relative_coordinates.sum(-1))

        self.qkv = nn.Linear(in_features=channels, out_features=channels*3, bias=qkv_bias)
        self.attention_drop = nn.Dropout(p=attention_drop)
        self.projection = nn.Linear(in_features=channels, out_features=channels)
        self.projection_drop = nn.Dropout(p=projection_drop)

        trunc_normal_(self.position_bias_table, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def calculate_coordinates(self):
        coordinates_height = torch.arange(self.window_size[0])
        coordinates_width = torch.arange(self.window_size[1])
        coordinates = torch.stack(torch.meshgrid([coordinates_height, coordinates_width], indexing="ij"))
        coordinates = torch.flatten(coordinates, 1)

        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[:, :, 0] += self.window_size[0] - 1
        relative_coordinates[:, :, 1] += self.window_size[1] - 1
        relative_coordinates[:, :, 0] *= 2 * self.window_size[1] - 1

        return relative_coordinates

    def forward(self, x, mask=None):
        batch, token, channel = x.shape # shape(B, N, C)
        qkv = self.qkv(x).reshape(batch, token, 3, self.num_heads, channel // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query = query * self.scale
        attention = (query @ key.transpose(-2, -1))

        relative_position_bias = self.position_bias_table[self.relative_coordinates.view(-1)].view(token, token, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            size = mask.shape[0]
            attention = attention.view(-1, size, self.num_heads, token, token) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, token, token)

        x = (attention @ value).transpose(1, 2).reshape(batch, token, channel)
        x = self.projection(x)
        x = self.projection_drop(x)

        return x


class WindowAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
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
        self.channels = channels
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.attention_norm = norm(channels)
        self.attention = WindowAttentionLayer(
            channels=channels, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attention_drop=attention_drop, projection_drop=projection_drop
        )

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        self.head_norm = norm(channels)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=channels, out_features=int(channels * mlp_ratio)),
            activation(),
            nn.Dropout(p=projection_drop),
            nn.Linear(in_features=int(channels * mlp_ratio), out_features=channels),
            nn.Dropout(p=projection_drop)
        )

        if self.shift_size > 0:
            height, width = self.input_resolution
            image_mask = torch.zeros((1, height, width, 1))
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )

            count = 0

            for _height in height_slices:
                for _width in width_slices:
                    image_mask[:, _height, _width, :] = count
                    count += 1

            mask_windows = self.window_partition(image_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask = attention_mask.masked_fill(
                attention_mask != 0, float(-100.0)
            ).masked_fill(attention_mask == 0, float(0.0))
        else:
            attention_mask = None

        self.register_buffer("attention_mask", attention_mask)

    def window_partition(self, x):
        batch, height, width, channel = x.shape
        x = x.view(batch, height // self.window_size, self.window_size, width // self.window_size, self.window_size, channel)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, channel)

        return windows

    def window_reverse(self, windows, height, width):
        batch = int(windows.shape[0] / (height * width / self.window_size / self.window_size))
        x = windows.view(batch, height // self.window_size, width // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)

        return x

    def forward(self, x):                       # TODO Check the shape
        height, width = self.input_resolution
        batch, token, channel = x.shape

        assert token == height * width, f"Input feature has wrong size token: {token} != height:({height}) X (width:{width})."

        shortcut = x
        x = self.attention_norm(x)
        x = x.view(batch, height, width, channel)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channel)

        attention_windows = self.attention(x_windows, mask=self.attention_mask)
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, channel)

        shifted_x = self.window_reverse(attention_windows, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(batch, height * width, channel)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.head_norm(x)))

        return x

    def flops(self):
        flops = 0
        height, width = self.input_resolution
        flops += self.channels * height * width

        num_windows = height * height / self.window_size / self.window_size
        flops += num_windows * self.attention.flops(self.window_size, self.window_size)
        flops += 2 * height * width * self.channels * self.channels * self.mlp_ratio
        flops += self.channels * height * width

        return flops
