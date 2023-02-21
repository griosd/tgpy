import torch
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import torch.nn as nn
from .tensor import to_tensor

class Constant(nn.Module):
    def __init__(self, tensor, device=None, *args, **kwargs):
        """
        Creates the constant module

        :param tensor: a numpy array, list or torch Tensor, the constant. 
        :param device: a string, the device to create de tensor.
        """
        super(Constant, self).__init__(*args, **kwargs)
        if isinstance(tensor, torch.Tensor):
            self.tensor = tensor
        else:
            self.tensor = to_tensor(tensor, device=device)

    def forward(self):
        return self.tensor
