import torch.nn as nn

from transformers import SwinModel


class SwinEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path="microsoft/swin-tiny-patch4-window7-224",
        output_hidden_states=False
    ):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=output_hidden_states
        )
        self.hidden_states = output_hidden_states

    def set_requires_grad(self, layers=None):
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        if layers is None:
            return

        if isinstance(layers, (list, tuple)):
            layers = [f"encoder.layers.{index}." for index in layers]

            for layer, param in self.backbone.named_parameters():
                if any(substring in layer for substring in layers):
                    param.requires_grad_(True)

    def forward(self, x):
        if self.hidden_states:
            hidden_states = self.backbone(x).hidden_states[:4]
            hidden_states = list(hidden_states)[::-1]

            return hidden_states
        else:
            return self.backbone(x)
