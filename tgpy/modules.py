import torch
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import torch.nn as nn

class Constant(nn.Module):
    def __init__(self, tensor, device, *args, **kwargs) -> None:
        """
        _summary_

        :param tensor: _description_
        :param device: _description_
        """
        super(Constant, self).__init__(*args, **kwargs)
        # if tensor, just save, if float, convert to tensor, if array, convert to tensor.
        if isinstance(tensor, torch.Tensor):
            self.tensor = tensor
        else:
            try:
                self.tensor = torch.ones(1, device=device) * tensor
            except:
                pass
    def forward(self):
        return self.tensor
