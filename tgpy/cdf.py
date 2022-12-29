import torch
import torch.nn as nn
from .tensor import one, eps4, two, zero
from torch.special import gammainc
from .tensor import gammaincinv, betainc, betaincinv


class CDF(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CDF, self).__init__(*args, **kwargs)

    @staticmethod
    def F(x, n=None):
        pass

    @staticmethod
    def Q(y, n=None):
        pass


class FisherSnedecor(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CDF, self).__init__(*args, **kwargs)

    @staticmethod
    def F(x, n, v):
        dx1 = (n * x) / (n * x + v)
        y = betainc(n / 2, v / 2, dx1)
        return y

    @staticmethod
    def Q(y, n, v):
        dx1 = betaincinv(n / 2, v / 2, y)
        x = v * dx1 / (n - n * dx1)
        return x


class NormGaussian(CDF):
    """2-Norm of a Gaussian based on Chi"""
    def __init__(self, *args, **kwargs):
        super(NormGaussian, self).__init__(*args, **kwargs)

    def F(self, x, n=1):
        n = torch.as_tensor(n, dtype=x.dtype, device=x.device)
        return gammainc(n/2, x**2/2)

    def Q(self, y, n=1):
        n = torch.as_tensor(n, dtype=y.dtype, device=y.device)
        return (2*gammaincinv(n/2, y))**0.5


class NormStudentT(CDF):
    """2-Norm of a Student-t based on FisherSnedecor (scaled sqrt)"""
    def __init__(self, v=3, *args, **kwargs):
        super(NormStudentT, self).__init__(*args, **kwargs)
        self.v = v

    @property
    def _v(self):
        return torch.as_tensor(self.v + 2.0, dtype=torch.float32, device='cuda')

    def F(self, x, n=1):
        n = torch.as_tensor(n, dtype=x.dtype, device=x.device)
        return FisherSnedecor.F((x ** 2) / n, n, self._v)

    def Q(self, y, n=1):
        n = torch.as_tensor(n, dtype=y.dtype, device=y.device)
        return (n * FisherSnedecor.Q(y, n, self._v)) ** 0.5

    def posterior_F(self, x, n=1, n_obs=0, norm_y_obs=0):
        n, n_obs = torch.as_tensor([n, n_obs], dtype=x.dtype, device=x.device)
        factor = (self._v + n_obs)/(self._v + norm_y_obs ** 2)
        return FisherSnedecor.F((x ** 2) * factor / n, n, self._v+n_obs)

    def posterior_Q(self, y, n=1, n_obs=0, norm_y_obs=0):
        n, n_obs = torch.as_tensor([n, n_obs], dtype=y.dtype, device=y.device)
        # n = torch.as_tensor(n, dtype=y.dtype, device=y.device)
        factor_inv = (self._v + norm_y_obs ** 2) / (self._v + n_obs)
        return (n * FisherSnedecor.Q(y, n, self._v+n_obs) * factor_inv) ** 0.5


