import torch
import torch.nn as nn


class TgLocation(nn.Module):
    pass


class Constant(TgLocation):
    def __init__(self, c=None, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        self.c = c

    def forward(self, x):
        return self.c


class Linear(TgLocation):
    def __init__(self, weight=None, offset=None, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.weight = weight
        self.offset = offset

    def forward(self, x):
        return self.offset+torch.mul(self.weight, x)
