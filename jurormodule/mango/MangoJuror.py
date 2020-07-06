import torch
from typing import Iterator
from torch.nn import Parameter

class MangoJuror(torch.nn.Module):
    def __init__(self, backbonenet):
        super(MangoJuror, self).__init__()
        self.backbonenet = backbonenet
        self.l1 = torch.nn.Linear(1000,3)

    def forward(self, x):
        return torch.nn.functional.softmax(self.l1(self.backbonenet(x)), 1)


    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]:
        if self.training:
            return super(MangoJuror, self).parameters(recurse)
        else:
            return []
