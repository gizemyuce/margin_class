import torch
import torch.nn as nn
from torch.nn import Linear, Sequential

class ConvNet(Sequential):
    """Same architecture as Byrd & Lipton 2017 on CIFAR10
    Args:
        output_size: dimensionality of final output, usually the number of classes
    """

    def __init__(self):
        layers = [
            torch.nn.Conv2d(3, 64, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),   #(1, -1),
            torch.nn.Linear(1152, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        ]
        super().__init__(*layers)

    @property
    def linear_output(self):
        return list(self.modules())[0][-1]


class ConvNet_binary(Sequential):
    """Same architecture as Byrd & Lipton 2017 on CIFAR10
    Args:
        output_size: dimensionality of final output, usually the number of classes
    """

    def __init__(self):
        layers = [
            torch.nn.Conv2d(1, 64, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),   #(1, -1),
            torch.nn.Linear(1152, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        ]
        super().__init__(*layers)

    @property
    def linear_output(self):
        return list(self.modules())[0][-1]
