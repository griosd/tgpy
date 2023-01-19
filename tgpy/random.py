import math
import torch
import torch.nn as nn
from .tensor import cholesky, _device, DataTensor
from .prior import TgPrior
from .cdf import NormGaussian

log2pi = math.log(2 * math.pi)


class TgTransport(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TgTransport, self).__init__(*args, **kwargs)

    def forward(self, x, h, noise=False):
        """ $T_{t}(x)$ """
        pass

    def inverse(self, x, y, noise=True):
        """ $T_{t}^{-1}(y)$ """
        pass

    def prior(self, x, nsamples=1, generator=None, noise=False):
        """ $T_{t}(x)$ """
        return self.forward(x, generator.prior(x, nsamples=nsamples), noise=noise)

    def posterior(self, x, h, obs_x, obs_y, generator, noise=False):
        """ $T_{t}(x)$ """
        pass

    def logdetgradinv(self, x, y, sy=None):
        pass


class MarginalTransport(TgTransport):
    def __init__(self, mapping, *args, **kwargs):
        super(MarginalTransport, self).__init__(*args, **kwargs)
        self.mapping = mapping

    def forward(self, x, h, noise=None):
        # print(self.name, 'forward', noise)
        return self.mapping(x, h)

    def inverse(self, x, y, noise=None):
        # print(self.name, 'inverse', noise)
        return self.mapping.inverse(x, y)

    def posterior(self, x, h, obs_t=None, obs_y=None, generator=None, noise=False, noise_cross=False):
        """ $T_{t}(x)$ """
        return self.forward(x, h, noise=noise)

    def logdetgradinv(self, x, y, sy=None):
        return self.mapping.log_gradient_inverse(x, y).nansum(dim=[1, 2])


class CovarianceTransport(TgTransport):
    def __init__(self, kernel, noise=None, use_cpu=False, *args, **kwargs):
        super(CovarianceTransport, self).__init__(*args, **kwargs)
        self.kernel = kernel
        self.noise = noise
        self.cholesky_cache = None
        self.use_cpu = use_cpu

    def kernel_noise(self, t1, t2=None):
        if self.noise is None:
            return self.kernel(t1, t2)
        else:
            return self.kernel(t1, t2) + self.noise(t1, t2)

    def forward(self, x, h, noise=False):
        # print(self.name, self.kernel, 'forward', noise)
        """$cho y$"""
        if noise:
            kernel = self.kernel_noise
        else:
            kernel = self.kernel
        return torch.matmul(cholesky(kernel(x)), h)

    def inverse(self, x, y, noise=True):
        # print(self.name, self.kernel, 'inverse', noise)
        """$cho^{-1}y$"""
        if noise:
            kernel = self.kernel_noise
        else:
            kernel = self.kernel
        self.cholesky_cache = cholesky(kernel(x))
        return torch.linalg.solve_triangular(self.cholesky_cache, y, upper=False)

    def posterior(self, x, h, obs_x, obs_y, generator=None, noise=False, noise_cross=False):
        # print(self.name, self.kernel, 'posterior', noise)
        if noise:
            kernel = self.kernel_noise
            kernel_cross = self.kernel_noise if noise_cross else self.kernel
        else:
            kernel = self.kernel
            kernel_cross = kernel
        device = 'cpu' if self.use_cpu else obs_y.device
        cross = kernel_cross(obs_x, x).to(device)
        cho = cholesky(self.kernel_noise(obs_x).to(device))
        mean = cross.transpose(-2, -1).matmul(torch.cholesky_solve(obs_y.to(device), cho))
        cross = torch.linalg.solve_triangular(cho, cross, upper=False)
        cho = -cross.transpose(-2, -1).matmul(cross)
        del cross
        h = torch.matmul(cholesky(cho.add(kernel(x).to(device))), h.to(device)).add(mean)
        del cho
        del mean
        return h.to(obs_y.device)

    def logdetgradinv(self, x, y=None, sy=None, noise=True):
        if self.cholesky_cache is None:
            if noise:
                self.cholesky_cache = cholesky(self.kernel_noise(x))
            else:
                self.cholesky_cache = cholesky(self.kernel(x))
        return -torch.diagonal(self.cholesky_cache, dim1=1, dim2=2).log().nansum(dim=1)


class Norm2Transport(TgTransport):
    """ y = Q(F(|x|)) x / |x| """
    def __init__(self, obj, *args, **kwargs):
        super(Norm2Transport, self).__init__(*args, **kwargs)
        self.obj = obj
        ref = NormGaussian()
        self.ref = ref

    def norm_y(self, norm_x, n=None):
        return self.obj.Q(self.ref.F(norm_x, n), n)

    def norm_x(self, norm_y, n=None):
        return self.ref.Q(self.obj.F(norm_y, n), n)

    def forward(self, t, x, noise=None):
        normx = x.norm(dim=-2)
        return self.norm_y(normx, n=t.shape[0]) * x / normx

    def inverse(self, t, y, noise=None):
        normy = y.norm(dim=-2)[:, :, None]
        scale = self.norm_x(normy, n=t.shape[0]) / normy
        return scale * y

    def posterior_norm_y(self, norm_x, n=None, n_obs=None, norm_y_obs=None):
        return self.obj.posterior_Q(self.ref.F(norm_x, n), n, n_obs, norm_y_obs)

    def posterior(self, t, x, obs_t, obs_y, generator=None, noise=False, *args, **kwargs):
        normx = x.norm(dim=-2)
        normy_obs = obs_y.norm(dim=-2)
        post_normy = self.posterior_norm_y(normx, n=t.shape[0], n_obs=obs_t.shape[0], norm_y_obs=normy_obs)
        return post_normy * x / normx

    def logdetgradinv(self, t, y, sy=None, eps=1e-4):
        normy = y.norm(dim=-2)[:, :, None]
        normx = self.norm_x(normy, n=t.shape[0])
        normy = self.norm_y(normx, n=t.shape[0])

        scale = normy/normx
        dalpha = (self.norm_x(normy * (1 + eps)+eps,
                              n=t.shape[0]) - self.norm_x(normy * (1 - eps)-eps, n=t.shape[0])) / ((normy+1) * 2*eps)
        # print(scale)
        # print(dalpha)
        scale[dalpha[:, 0, 0] != dalpha[:, 0, 0], 0, 0] = 1e-10
        dalpha[dalpha[:, 0, 0] != dalpha[:, 0, 0], 0, 0] = 1e-10
        # print(-(t.shape[0]-1)*scale[:, 0, 0].log() + dalpha[:, 0, 0].log())
        return -(t.shape[0]-1)*scale[:, 0, 0].log() + dalpha[:, 0, 0].log()


class TgRandomField(nn.Module):

    def __init__(self, dt: DataTensor = None, *args, **kwargs):
        super(TgRandomField, self).__init__(*args, **kwargs)
        self.dt = dt
        self.obs_index = None
        self.obs_x = None
        self.obs_y = None
        self.is_iid = False
        self.description = {}
        self._priors_dict = None
        self._priors_list = None

    def obs(self, index=None, x=None, y=None):
        self.obs_index = index
        self.obs_x = x if self.obs_index is None else self.dt.tensor_inputs(index)
        self.obs_y = y if self.obs_index is None else self.dt.tensor_outputs(index)

    def logp(self, h):
        pass

    def prior(self, x, nsamples=1, noise=False):
        pass

    def posterior(self, x, nsamples=1, obs_t=None, obs_y=None, noise=False):
        pass

    def sample(self, x, n=1):
        if self.obs_x is None:
            return self.prior(x, n)
        else:
            return self.posterior(x, n)

    def describe(self, title=None, x=None, y=None, text=None):
        """
        Set descriptions to the model
        Args:
            title (str): The title of the process
            x (str):the label for the dependient variable
            y (str):the label for the independient variable
            text (str): aditional text if necessary.
        """

        if title is not None:
            self.description['title'] = title
        if title is not None:
            self.description['x'] = x
        if title is not None:
            self.description['y'] = y
        if title is not None:
            self.description['text'] = text

    @property
    def priors(self):
        if self._priors_dict is None:
            self._priors_dict = {k.name: k for k in self.modules() if isinstance(k, TgPrior)}
        return self._priors_dict

    @property
    def priors_list(self):
        if self._priors_list is None:
            self._priors_list = list(self.priors.values())
        return self._priors_list

    @property
    def nparams(self):
        return self.priors_list[0].dim

    def logp_priors(self):
        return sum([p.logp() for p in self.priors.values()])

    def clamp(self):
        _ = [p.clamp() for p in self.priors.values()]

    def clamp_grad(self):
        _ = [p.clamp_grad() for p in self.priors.values()]

    def sample_priors(self):
        _ = [p.sample_params() for p in self.priors.values()]

    def plot_priors(self, *args, **kwargs):
        for prior in sorted(self.priors.keys()):
            self.priors[prior].plot(*args, **kwargs)


class GWNP(TgRandomField):
    def __init__(self, *args, **kwargs):
        super(GWNP, self).__init__(*args, **kwargs)
        self.dtype = torch.float32
        self.is_iid = True

    def prior(self, x, nsamples=1, noise=False):
        return torch.empty((x.shape[0], nsamples), dtype=self.dtype, device=_device).normal_()

    def posterior(self, x, nsamples=1, obs_t=None, obs_y=None, noise=False):
        return torch.empty((x.shape[0], nsamples), dtype=self.dtype, device=_device).normal_()

    def logp(self, h):
        """x.shape = (nparams, nobs, noutput)"""
        return -0.5 * (h.shape[1] * log2pi + (h ** 2).sum(dim=1)).sum(dim=1)


class TP(TgRandomField):
    def __init__(self, generator, transport, annealing=1, annealing_prior=1, *args, **kwargs):
        super(TP, self).__init__(*args, **kwargs)
        self.generator = generator
        if isinstance(transport, TgTransport):
            self.transport = nn.ModuleList([transport])
        else:
            self.transport = nn.ModuleList(transport)
        self.annealing = annealing
        self.annealing_prior = annealing_prior

    def forward(self, x, h, noise=False):
        hi = h
        for T in self.transport:
            hi = T.forward(x, hi, noise=noise)
        return hi

    def inverse(self, x, y, noise=True, return_inv=True, return_list=False):
        list_inv_y = [y]
        for T in reversed(self.transport[1:]):
            list_inv_y.append(T.inverse(x, list_inv_y[-1], noise=noise))
        if return_inv:
            list_inv_y.append(self.transport[0].inverse(x, list_inv_y[-1], noise=noise))
            yi = list_inv_y[-1]
            if return_list:
                list_inv_y.reverse()
                return yi, list_inv_y
            else:
                return yi
        elif return_list:
            list_inv_y.reverse()
            return list_inv_y

    def logdetgradinv(self, x, list_obs_y):
        r = self.transport[0].logdetgradinv(x, y=list_obs_y[1], sy=list_obs_y[0])
        # print(self.transport[0], self.transport[0].logdetgradinv(t, y=list_obs_y[1], sy=list_obs_y[0]))
        for i in range(1, len(self.transport)):
            r += self.transport[i].logdetgradinv(x, y=list_obs_y[i + 1], sy=list_obs_y[i])
            # print(self.transport[i], self.transport[i].logdetgradinv(t, y=list_obs_y[i+1], sy=list_obs_y[i]))
        return r

    @property
    def obs_h(self):
        return self.inverse(self.obs_x, self.obs_y, noise=True)

    def prior(self, x, nsamples=1, noise=False):
        return self.forward(x, self.generator.prior(x, nsamples=nsamples), noise=noise)

    def posterior(self, x, nsamples=1, obs_x=None, obs_y=None, noise=False, noise_cross=False):
        if obs_x is None:
            obs_x = self.obs_x
        if obs_y is None:
            obs_y = self.obs_y
        list_obs_y = self.inverse(obs_x, obs_y, return_inv=False, return_list=True, noise=True)
        hi = self.generator.posterior(x, nsamples=nsamples)
        for i, T in enumerate(self.transport):
            hi = T.posterior(x, hi, obs_x, list_obs_y[i], generator=self.generator, noise=noise,
                             noise_cross=noise_cross)
        return hi

    def sample(self, inputs, nsamples=100, noise=False, noise_cross=False):
        if len(inputs.shape) == 1:
            x = self.dt.tensor_inputs(inputs)
            columns = self.dt.original_inputs(inputs).index
        else:
            x = self.dt.original_to_tensor_inputs(inputs)
            columns = inputs.index
        samples = self.posterior(x, nsamples=nsamples, noise=noise, noise_cross=noise_cross)
        samples = samples.transpose(-1, -2).reshape(-1, samples.shape[1])
        samples = self.dt.tensor_to_original_outputs(samples)
        samples.columns = columns
        return samples

    def predict(self, inputs, quantiles=0.2, median=True, mean=True, nsamples=100, samples=False, noise=False,
                noise_cross=False):
        """
        Predicts with the transport process.

        :param inputs: _description_
        :param quantiles: a float, between [0, 1] quantile to create confidence interval, defaults to 0.2.
        :param median: a bool, if True the prediction will be the median of the samples, defaults to True.
        :param mean: a bool, if true the prediction will be the median of the samples, overrides 'median' arg,
            defaults to True.
        :param nsamples: an int, number of samples, defaults to 100.
        :param samples: a bool, if True it returns the samples alongside the dataframe with results,  defaults to False.
        :param noise: a bool, if True the samples will include noise , defaults to False.
        :param noise_cross: a bool, if True samples assuming noisy observations, defaults to False.

        :return: a pandas DataFrame, with statistic (median or mean) and confidence intervals, if 'samples' if True
            also return the process samples in DataFrame form. 
        """
        _samples = self.sample(inputs, nsamples=nsamples, noise=noise, noise_cross=noise_cross)
        if len(inputs.shape) == 1:
            pred = self.dt.original(inputs)
        else:
            pred = inputs.copy()
        if median:
            pred['Median'] = _samples.median()
            if self.dt.outputs[0] in pred.columns:
                pred['Median Error'] = pred[self.dt.outputs[0]] - pred['Median']
        if mean:
            pred['Mean'] = _samples.mean()
            if self.dt.outputs[0] in pred.columns:
                pred['Mean Error'] = pred[self.dt.outputs[0]] - pred['Mean']
        pred['Quantile {}'.format(int(100 * (1 - quantiles)))] = _samples.quantile(1 - quantiles)
        pred['Quantile {}'.format(int(100 * quantiles))] = _samples.quantile(quantiles)
        pred['Interval'] = _samples.quantile(1 - quantiles) - _samples.quantile(quantiles)
        pred['Certainty'] = (pred['Interval'].max() - pred['Interval'])/pred['Interval'].max()
        if samples:
            return pred, _samples
        else:
            return pred

    def logp_likelihood(self, x=None, y=None, index=Ellipsis):
        if x is None:
            x = self.obs_x[index]
        if y is None:
            y = self.obs_y[index]
        inverse_y, list_obs_y = self.inverse(x, y, return_inv=True, return_list=True)
        return self.generator.logp(inverse_y) + self.logdetgradinv(x, list_obs_y=list_obs_y)

    def logp(self, x=None, y=None, index=Ellipsis):
        r = 0
        if self.annealing:
            r += self.logp_likelihood(x=x, y=y, index=index)
        if self.annealing_prior:
            r += self.logp_priors()
        return r


class TGP(TP):
    def __init__(self, transport, *args, **kwargs):
        super(TGP, self).__init__(generator=GWNP(), transport=transport, *args, **kwargs)
