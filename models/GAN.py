from typing import Any

import torch
import torch.nn as nn

import torchvision


class GAN_Gen(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super().__init__()
        print("INIT GAN GENERATOR")
        self.main_module = nn.Sequential(
            # z latent vector is 100x1 Normal dist
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Tanh()
        )

    def forward(self, x):
        # x = torch.flatten(x, start_dim=2, end_dim=3)
        return self.main_module(x)


class GAN_Des(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super().__init__()
        print("INIT GAN DISCRIMINATOR")
        self.main_module = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = torch.flatten(x, start_dim=2, end_dim=3)
        x = self.main_module(x)
        return x
