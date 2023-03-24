import torch
import torch.nn as nn
from torch.distributions.chi2 import Chi2


class TgRadial(nn.Module):
    """
    phi(||x||) x, phi(r) = alpha(r) / r.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(TgRadial, self).__init__(*args, **kwargs)

    def forward(self, t, x):
        raise NotImplementedError

    def inverse(self, t, y):
        raise NotImplementedError

    def log_gradient_inverse(self, t, y):
        # log det inverse S_t (y) = (n - 1) log [alpha^(-1)||y||]  - log [alpha' (alpha^{-1} (||y||))]
        pass

    def __matmul__(self, other):
        """
        Implements the __matmul__ method, to compose two Radial Maps.

        :param other: another Marginal.

        :return: the same Composition object with updated number of operations.
        """
        return RadialComposition(self, other)


class RadialComposition(TgRadial):

    def __init__(self, m0, m1, *args, **kwargs) -> None:
        super(RadialComposition).__init__(*args, **kwargs)
        self.m0 = m0
        self.m1 = m1
    
    def forward(self, t, x):
        pass

class DummyRadial(TgRadial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # phi = alpha(r) / r, in this case alpha(r) = r, so phi(r) 
        # 

    def forward(self, t, x):
        # apply mapping phi(||x||) x taking into account the norm
        return x

    def inverse(self, t, y):
        return y

    def log_gradient_inverse(self, t, y):
        # log det inverse S_t (y) = (n - 1) log [alpha^(-1)||y||]  - log [alpha' (alpha^{-1} (||y||)) ]
        pass


class ChiSquaredInv(TgRadial):
    def __init__(self, *args, **kwargs):
        super(TgRadial, self).__init__()
        # duda, degrees of freedom equals length (t or y) ????

    def forward(self, t, x):
        degrees = len(x.squeeze())
        distribution = Chi2(torch.torch.tensor[degrees])
        #  obtain inverse of  cdf

        return 

    def inverse(self, t, y):
        # return cdf
        return 

class StudentT(TgRadial):
    def __init__(self, *args, **kwargs) -> None:
        # TODO: add student-t parameters
        super(TgRadial, self).__init__()
        self.radial_map = None # TODO(ale) given the parameters create the atribute with the map

    def forward(self, x):
        # implement particular case forward.
        pass
