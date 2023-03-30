import torch
import torch.nn as nn
from scipy.stats import chi2


class TgRadial(nn.Module):
    # phi(||x||) x, phi(r) = alpha(r) / r.
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
        return RadialComposition(self, other)


class RadialComposition(TgRadial):
    def __init__(self, m0, m1, *args, **kwargs) -> None:
        super(RadialComposition).__init__(*args, **kwargs)
        self.m0 = m0
        self.m1 = m1
    
    def forward(self, t, x):
        self.m0(t, self.m1(t, x))

    def inverse(self, t, y):
        return self.m1.inverse(t, self.m0.inverse(t, y))

    def log_gradient_inverse(self, t, y):
        return self.m1.log_gradient_inverse(t, self.m0.inverse(t, y)) + self.m0.log_gradient_inverse(t, y)


class DummyRadial(TgRadial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # phi = alpha(r) / r, in this case alpha(r) = r, so phi(r) 

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
        degrees = len(x.squeeze()) # degrees of freedom
        #  obtain inverse of  cdf
        return chi2.ppf(x, degrees)

    def inverse(self, t, y):
        # return cdf
        degrees = len(y.squeeze()) 

        return chi2.cdf(y, degrees)

class StudentT(TgRadial):
    def __init__(self, *args, **kwargs) -> None:
        # TODO: add student-t parameters
        super(TgRadial, self).__init__()
        self.radial_map = None # TODO(ale) given the parameters create the atribute with the map

    def forward(self, x):
        # implement particular case forward.
        pass
