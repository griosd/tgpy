import torch
import math
from torch import nn
from .metric import L0, L1, L2, MinimumSum, Diff, Prod


class TgKernel(nn.Module):
    def __init__(self, metric=None, inputs=Ellipsis):
        super(TgKernel, self).__init__()
        self.metric = metric
        self.inputs = inputs

    def __mul__(self, other):
        if issubclass(type(other), TgKernel):
            return KernelProd(self, other)
        else:
            return KernelScale(self, other)
    __imul__ = __mul__

    def __rmul__(self, other):
        if issubclass(type(other), TgKernel):
            return KernelProd(other, self)
        else:
            return KernelScale(self, other)

    def __add__(self, other):
        if issubclass(type(other), TgKernel):
            return KernelSum(self, other)
        else:
            return KernelShift(self, other)
    __iadd__ = __add__

    def __radd__(self, other):
        if issubclass(type(other), TgKernel):
            return KernelSum(other, self)
        else:
            return KernelShift(self, other)


class KernelOperation(TgKernel):
    def __init__(self, _k: TgKernel, _element, *args, **kwargs):
        super(TgKernel, self).__init__(*args, **kwargs)
        self.kernel = _k
        self.element = _element


class KernelScale(KernelOperation):
    def forward(self, x1, x2=None):
        return self.element * self.kernel.forward(x1, x2)


class KernelShift(KernelOperation):
    def forward(self, x1, x2=None):
        return self.element + self.kernel.forward(x1, x2)


class KernelComposition(TgKernel):
    def __init__(self,  _k1: TgKernel, _k2: TgKernel, *args, **kwargs):
        super(TgKernel, self).__init__(*args, **kwargs)
        self.kernel1 = _k1
        self.kernel2 = _k2


class KernelProd(KernelComposition):
    def forward(self, x1, x2=None):
        return self.kernel1.forward(x1, x2) * self.kernel2.forward(x1, x2)


class KernelSum(KernelComposition):
    def forward(self, x1, x2=None):
        return self.kernel1.forward(x1, x2) + self.kernel2.forward(x1, x2)

# Non stationary kernels

class BW(TgKernel):
    def __init__(self, relevance, inputs=Ellipsis):
        super(BW, self).__init__(inputs=inputs)
        self.relevance = relevance
        self.metric = MinimumSum(self.relevance, inputs=inputs)

    def forward(self, x1, x2=None):
        return self.metric(x1, x2)


class LIN(TgKernel):
    def __init__(self, relevance, inputs=Ellipsis):
        super(LIN, self).__init__(inputs=inputs)
        self.relevance = relevance
        self.metric = Prod(self.relevance, inputs=inputs)

    def forward(self, x1, x2=None):
        return self.metric(x1, x2)


class POL(TgKernel):
    def __init__(self, relevance, power, inputs=Ellipsis):
        super(POL, self).__init__(inputs=inputs)
        self.relevance = relevance
        self.power = power
        self.metric = Prod(self.relevance, inputs=inputs)

    def forward(self, x1, x2=None):
        f = self.metric(x1, x2)
        return torch.sign(f) * (f.abs() ** self.power()[:, :, None])

# Stationary kernels

class RQ(TgKernel):
    def __init__(self, var, relevance, freedom, inputs=Ellipsis):
        super(RQ, self).__init__(inputs=inputs)
        self.var = var
        self.relevance = relevance
        self.freedom = freedom
        self.metric = L2(self.relevance, inputs=inputs)

    def forward(self, x1, x2=None):
        freedom = self.freedom()[:, :, None]
        return self.var()[:, :, None] * (1 + self.metric(x1, x2)/freedom).pow(-freedom)


class SE(TgKernel):
    def __init__(self, var, relevance, inputs=Ellipsis):
        super(SE, self).__init__(inputs=inputs)
        self.var = var
        self.relevance = relevance
        self.metric = L2(self.relevance, inputs=inputs)

    def forward(self, x1, x2=None):
        return self.var()[:, :, None] * (-self.metric(x1, x2)).exp()


class OU(TgKernel):
    def __init__(self, var, relevance, inputs=Ellipsis):
        super(OU, self).__init__(inputs=inputs)
        self.var = var
        self.relevance = relevance
        self.metric = L1(self.relevance, inputs=inputs)

    def forward(self, x1, x2=None):
        return self.var()[:, :, None] * (-self.metric(x1, x2)).exp()

# Periodic Kernels

class SINC(TgKernel):
    """SINC periodic kernel"""
    def __init__(self, var, relevance, period, inputs=Ellipsis):
        super(SINC, self).__init__(inputs=inputs)
        self.var = var
        self.relevance = relevance
        self.period = period
        self.metric = Diff(inputs=inputs)

    def forward(self, x1, x2=None):
        # note: torch sinc function is sin(pi x) / (pi x), that means pi multiplication is included.
        relevance = self.relevance()[:, None, :]
        period = self.period()[:, None, :]
        sinc_term = torch.sinc(self.metric(x1, x2) / relevance)
        cos_term = torch.cos(2 * math.pi * self.metric(x1, x2) / period)

        return self.var()[:, :, None] * sinc_term * cos_term


class SM(TgKernel):
    """Spectral mixture kernel"""
    def __init__(self, var, relevance, period, inputs=Ellipsis):
        super(SM, self).__init__(inputs=inputs)
        self.var = var
        self.relevance = relevance
        self.period = period
        self.metric = Diff(inputs=inputs)

    def forward(self, x1, x2=None):
        relevance = self.relevance()[:, None, :]
        exp_term = (-(self.metric(x1, x2) / relevance).pow(2)).exp()
        cos_term = torch.cos((2 * math.pi * self.metric(x1, x2)) / self.period()[:, :, None])
        return self.var()[:, :, None] * exp_term * cos_term


class SIN(TgKernel):
    """Sin periodic kernel"""
    def __init__(self, var, relevance, period, inputs=Ellipsis):
        super(SIN, self).__init__(inputs=inputs)
        self.var = var
        self.relevance = relevance
        self.period = period
        self.metric = Diff(inputs=inputs)

    def forward(self, x1, x2=None):
        relevance = self.relevance()[:, None, :]
        period = self.period()[:, None, :]
        exp_arg = 2 * torch.sin(math.pi * self.metric(x1, x2) / period).pow(2) / (relevance ** 2)
        return self.var()[:, :, None] * (-exp_arg).exp()

# Misc Kernels

class WN(TgKernel):
    def __init__(self, var, inputs=Ellipsis):
        super(WN, self).__init__(inputs=inputs)
        self.var = var
        self.metric = L0(inputs=inputs)

    def forward(self, x1, x2=None):
        return self.var()[:, :, None] * self.metric(x1, x2)


class DummyKernel(TgKernel):
    def __init__(self, *args, inputs=Ellipsis):
        super(DummyKernel, self).__init__(inputs=inputs)
        for i, arg in enumerate(args):
            setattr(self, 'prior{}'.format(i), arg)
        self.metric = L0(inputs=inputs)

    def forward(self, x1, x2=None):
        return 0 * self.metric(x1, x2)
