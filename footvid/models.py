from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        resnet50 = models.resnet50(pretrained=True, progress=False)
        conv_layers = []
        for named_child in resnet50.named_children():
            conv_layers.append(named_child)
            if named_child[0] == "layer4":
                break

        self.resnet50_conv = nn.Sequential(OrderedDict(conv_layers))
        self.avgpool = resnet50.avgpool
        self.fc = nn.Linear(
            in_features=resnet50.fc.in_features,
            out_features=output_size,
            bias=True,
        )

        self._gradients = None

    @property
    def gradients(self) -> torch.Tensor:
        return self._gradients

    def set_gradients(self, gradients: torch.Tensor) -> None:
        self._gradients = gradients

    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet50_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet50_conv(x)
        if x.requires_grad:
            x.register_hook(self.set_gradients)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
