import torch
from typing import Iterator
from torch.nn import Parameter

class MangoJuror(torch.nn.Module):
    def __init__(self, backbonenet):
        super(MangoJuror, self).__init__()
        self.backbonenet = backbonenet
        self.l1 = torch.nn.Linear(1000, 5) # Three categories plus a confidence

    def forward(self, x):
        cateAndConfi = self.l1(self.backbonenet(x))
        cate = cateAndConfi[:, 0:3]
        confi = cateAndConfi[:, 3:]
        cateSoft = torch.nn.functional.softmax(cate, dim=1)
        confiSoft = torch.nn.functional.softmax(confi, dim=1)
        return torch.cat(cateSoft, confiSoft)
