import torch
from torch import nn
from .tensor import dim_index, dim_series, dim_gram


class TgMetric(nn.Module):
    def __init__(self, inputs=Ellipsis):
        super(TgMetric, self).__init__()
        self.inputs = inputs

    def m(self, x1, x2):
        """x1.shape = (1, dim_index1, dim_series)
           x2.shape = (dim_index2, 1, dim_series)"""
        pass

    def forward(self, x1, x2=None):
        """x1.shape = (dim_index1, dim_series)
           x2.shape = (dim_index2, dim_series)"""
        return self.m(*self.gram(x1, x2))

    def gram(self, x1, x2=None):
        """x1.shape = (dim_index1, dim_series)
           x2.shape = (dim_index2, dim_series)"""
        if x2 is None:
            x2 = x1
        return x1[:, None, self.inputs], x2[None, :, self.inputs]


class Diff(TgMetric):
    def __init__(self, inputs=Ellipsis):
        super(Diff, self).__init__(inputs=inputs)

    def m(self, x1, x2):
        return (x1 - x2)[None].sum(dim=-1)
        # return x1 - x2


class L0(TgMetric):
    def __init__(self, inputs=Ellipsis):
        super(L0, self).__init__(inputs=inputs)

    def m(self, x1, x2):
        dx = x1 - x2
        return (dx == 0).all(dim=-1) + 0.0
        # return (dx == 0).all(dim=-1)+0.0


class L1(TgMetric):
    def __init__(self, relevance, inputs=Ellipsis):
        super(L1, self).__init__(inputs=inputs)
        self.relevance = relevance #shape = (1, 1, dim_series)

    def m(self, x1, x2):
        dx = x1 - x2
        return (dx.abs()[None, :, :, :] / self.relevance()[:, None, None, :]).sum(dim=-1)
        # return (dx.abs()[None, :, :, :] / self.relevance[:, None, None, :]).sum(dim=-1)


class L2(TgMetric):
    def __init__(self, relevance, inputs=Ellipsis):
        super(L2, self).__init__(inputs=inputs)
        self.relevance = relevance #shape = (1, 1, dim_series)

    def m(self, x1, x2):
        dx = x1 - x2
        return (dx.abs()[None] / self.relevance()[:, None, None, :]).pow(2).sum(dim=-1)
        # return (dx[None, :, :, :] / self.relevance[:, None, None, :]).pow(2).sum(dim=-1)


class Prod(TgMetric):
    def __init__(self, relevance, inputs=Ellipsis):
        super(Prod, self).__init__(inputs=inputs)
        self.relevance = relevance #shape = (1, 1, dim_series)

    def m(self, x1, x2):
        return ((x1[None, :, :, :] / self.relevance()[:, None, None, :]) * x2[None, :, :, :]).sum(dim=-1)
        # return ((x1[None, :, :, :] / self.relevance[:, None, None, :]) * x2[None, :, :, :]).sum(dim=-1)


class MinimumProd(TgMetric):
    def __init__(self, relevance, inputs=Ellipsis):
        super(MinimumProd, self).__init__(inputs=inputs)
        self.relevance = relevance

    def m(self, x1, x2):
        return (torch.min(x1, x2)[None, :, :, :] / self.relevance()[:, None, None, :]).prod(dim=-1)
        # return (torch.min(x1, x2)[None, :, :, :] / self.relevance[:, None, None, :]).prod(dim=-1)


class MinimumSum(TgMetric):
    def __init__(self, relevance, inputs=Ellipsis):
        super(MinimumSum, self).__init__(inputs=inputs)
        self.relevance = relevance

    def m(self, x1, x2):
        return (torch.min(x1, x2)[None, :, :, :] / self.relevance()[:, None, None, :]).sum(dim=-1)
        # return (torch.min(x1, x2)[None, :, :, :] / self.relevance[:, None, None, :]).sum(dim=-1)
