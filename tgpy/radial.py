import torch
import torch.nn as nn


class TgRadial(nn.Module):
    """
    phi(||x||) x, phi(r) = alpha(r) / r.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(TgRadial, self).__init__(*args, **kwargs)

    def forward(self, t, x):
        raise NotImplementedError

    def log_gradient_inverse(self, t, y):
        # log det inverse S_t (y) = (n - 1) log [alpha^(-1)||y||]  - log [alpha' (alpha^{-1} (||y||)) ]
        pass

class CustomRadial(TgRadial):
    def __init__(self, radial_map, inverse_radial_map, norm='l2', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.radial_map = radial_map # alpha
        # obtiain phi = alpha(r) / r
        self.norm_type = norm

    def forward(self, t, x):
        # apply mapping phi(||x||) x taking into account the norm
        pass


class StudentTRadial(TgRadial):
    def __init__(self, *args, **kwargs) -> None:
        # TODO: add student-t parameters
        super(TgRadial, self).__init__()
        self.radial_map = None # TODO(ale) given the parameters create the atribute with the map

    def forward(self, x):
        # implement particular case forward.
        pass
