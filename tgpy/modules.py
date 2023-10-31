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

class Scaling(nn.Module):
    """
    This class implements the scaling of an NgModule by a tensor.

    :param m: an NgModule.
    :param scale: an tensor, the scale.
    :param args: arguments to be passed to the class NgModule.
    :param kwargs: keyword arguments to be passed to the class NgModule.
    """
    def __init__(self, module, scale, *args, **kwargs):
        # noinspection PyArgumentList
        super(Scaling, self).__init__(*args, **kwargs)
        self.module = module
        self.scale = scale

    def forward(self, *args, **kwargs):
        """
        Scales the stored NgModule or Addition objects.

        :param args: arguments to be passed to the NgModule class.
        :param kwargs: keyword arguments to be passed to the NgModule class.

        :return: the scaled version of the stored objects.
        """
        # noinspection PyArgumentList
        return self.module.forward(*args, **kwargs).mul(self.scale)
