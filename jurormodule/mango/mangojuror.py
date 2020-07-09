import torch
from typing import Iterator
from torch.nn import Parameter

class MangoJuror(torch.nn.Module):
    def __init__(self, backbonenet):
        super(MangoJuror, self).__init__()
        self.backbonenet = backbonenet
        self.l1 = torch.nn.Linear(1000, 4) # Three categories plus a confidence

    def forward(self, x):
        return torch.nn.functional.softmax(self.l1(self.backbonenet(x)), 1)
