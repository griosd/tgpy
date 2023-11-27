import math
import torch

import matplotlib.pyplot as plt
import torch.nn as nn

from .tensor import cholesky, _device, DataTensor
from .prior import TgPrior, TgPriorMarginal
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
        r = self.mapping.log_gradient_inverse(x, y)
        if len(r.shape) > 2:
            return r.nansum(dim=[1,2])
        else:
            return r.nansum(dim=[-1])


class RadialTransport(TgTransport):
    def __init__(self, radial, *args, **kwargs):
        super(RadialTransport).__init__(*args, **kwargs)
        self.radial = radial

    def forward(self, x, h, noise=False):
        """ $T_{t}(x)$ """
        return self.radial.forward(x, h)

    def inverse(self, x, y, noise=True):
        """ $T_{t}^{-1}(y)$ """

        return self.radial.inverse(x, y)

    def posterior(self, x, h, obs_x, obs_y, generator=None, noise=False, noise_cross=False):
        # eq from 6.1.3
        pass

    def logdetgradinv(self, x, y, sy=None):
        return self.radial.log_gradient_inverse(x, y).nansum(dim=[1, 2])


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


class TgRandomField(nn.Module):

    def __init__(self, dt: DataTensor = None, *args, **kwargs):
        super(TgRandomField, self).__init__(*args, **kwargs)
        self.dt = dt
        self.obs_index = None
        self.obs_x = None
        self.obs_y = None
        self.is_iid = False
        self.description = {}
        self._marginals_dict = None
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
    def marginals(self):
        if self._marginals_dict is None:
            self._marginals_dict = {k.name: k for k in self.modules() if isinstance(k, TgPriorMarginal)}
        return self._marginals_dict

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

    def plot_marginals(self, *args, **kwargs):
        for prior in sorted(self.marginals.keys()):
            self.marginals[prior].plot(*args, **kwargs)

    def l2error(self, *args, **kwargs):
        MSES = []
        for prior in sorted(self.priors.keys()):
            prior_error = self.priors[prior].l2error(*args, **kwargs)
            MSES.append(prior_error[:, None])
        return torch.concat(MSES, 1).transpose(0, 1)


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
        for i in range(1, len(self.transport)):
            r += self.transport[i].logdetgradinv(x, y=list_obs_y[i + 1], sy=list_obs_y[i])
        return r

    @property
    def obs_h(self):
        return self.inverse(self.obs_x, self.obs_y, noise=True)

    def prior(self, inputs, nsamples=1, noise=False):
        if len(inputs.shape) == 1:
            x = self.dt.tensor_inputs(inputs)
        else:
            x = self.dt.original_to_tensor_inputs(inputs)

        return self.forward(x, self.generator.prior(x, nsamples=nsamples), noise=noise)

    def posterior(self, x, nsamples=1, obs_x=None, obs_y=None, noise=False, noise_cross=False):
        if obs_x is None:
            obs_x = self.obs_x
        if obs_y is None:
            obs_y = self.obs_y
        list_obs_y = self.inverse(obs_x, obs_y+1e-4, return_inv=False, return_list=True, noise=True)
        hi = self.generator.posterior(x, nsamples=nsamples)
        for i, T in enumerate(self.transport):
            hi = T.posterior(x, hi, obs_x, list_obs_y[i], generator=self.generator, noise=noise,
                             noise_cross=noise_cross)
        return hi

    def latent(self, x, ntransport, nsamples=1, obs_x=None, obs_y=None, noise=False, noise_cross=False):
        if obs_x is None:
            obs_x = self.obs_x
        if obs_y is None:
            obs_y = self.obs_y
        list_obs_y = self.inverse(obs_x, obs_y, return_inv=False, return_list=True, noise=True)
        hi = self.generator.posterior(x, nsamples=nsamples)
        for i in range(ntransport):
            T = self.transport[i]
            hi = T.posterior(x, hi, obs_x, list_obs_y[i], generator=self.generator, noise=noise,
                             noise_cross=noise_cross)
        return hi

    def sample(self, inputs, nsamples=100, noise=False, noise_cross=False, latent=False, prior=False, ntransport=-1):
        if len(inputs.shape) == 1:
            x = self.dt.tensor_inputs(inputs)
            columns = self.dt.original_inputs(inputs).index
        else:
            x = self.dt.original_to_tensor_inputs(inputs)
            columns = inputs.index

        if prior:
            samples = self.prior(x, nsamples=nsamples, noise=noise)
        elif latent:
            N=len(self.transport)
            if ntransport==-1:
                ntransport=N
            if ntransport<=0 or type(ntransport)!=int:
                 raise ValueError('ntransport must be a positive int value.')
            if ntransport not in range(N+1):
                raise ValueError('TGP only have {0} transports, {1} out of range.'.format(N, ntransport))

            samples = self.latent(x, nsamples=nsamples, noise=noise, noise_cross=noise_cross,
                                  ntransport=ntransport)
        else:
            samples = self.posterior(x, nsamples=nsamples, noise=noise, noise_cross=noise_cross)
        samples = samples.transpose(-1, -2).reshape(-1, samples.shape[1])
        samples = self.dt.tensor_to_original_outputs(samples)
        samples.columns = columns
        return samples

    def predict(self, inputs, quantiles=0.2, median=True, mean=True, nsamples=100, samples=False, noise=False,
                noise_cross=False, latent=False, ntransport=-1, prior=False):
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
        :param latent: a bool, if True samples only until a latent transport of the TGP.
        :param ntransport: a positive int in range of the TGP tranports number, determines the last transport 
            where the function will predict.
        :return: a pandas DataFrame, with statistic (median or mean) and confidence intervals, if 'samples' if True
            also return the process samples in DataFrame form. 
        """

        if latent:
            _samples = self.sample(inputs, nsamples=nsamples, noise=noise, noise_cross=noise_cross,
                                   latent=True, ntransport=ntransport)
        elif prior:
            _samples = self.sample(inputs, nsamples=nsamples, noise=noise, prior=prior)
        else:
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

    def plot_predict(self,
                     title='GP',
                     x_label='$x$',
                     y_label='$y=f(x) + \eta$',
                     nsamples=100,
                     noise=False,
                     quantile=0.1,
                     prior=False,
                     valid_index=None,
                     return_pred=False,
                     return_samples=False,
                     plot_obs=True,
                     plot_CI=True,
                     plot_samples=False,
                     ylim_by_CI=True,
                     statistic='Mean',
                     samples_kwargs={'alpha':0.05},
                     CI_kwargs={'facecolor':'g', 'alpha':0.2, 'ls':'--'},
                     obs_kwargs={'c':'black', 'marker':'X', 's':100, 'label':'Observations'},
                     real_kwargs={'c':'black', 'ls':'--', 'lw':3, 'label':'Real Signal'},
                     pred_kwargs={'c':'g', 'lw':4, 'label':'Predicted Signal', 'alpha':1.0}):
        """
        Creates the prediction of a GP and plots it.

        :param title: a string, determines the title of the figure, defaults to 'GP'.
        :param x_label: a string, determines the x label of the figure, defaults to '$x$'.
        :param y_label: a string, determines the x label of the figure, defaults to $y=f(x) + \eta$'.
        :param nsamples: an int, number of samples, default to 100.- scaled to 100
        :param noise: a bool, if True the samples will include noise, defaults to False.
        :param quantile: a float, between [0, 1] quantile to create confidence interval, defaults to 0.1.
        :param valid_index: a numpy.ndarray, the index of values for validation, defaults to None.
        :param return_pred: a bool, if True, the function returns the DataFrame with the prediction, defaults to False.
        :param return_samples: a bool, if True, the function returns the DataFrame with the samples, defaults to False.
        :param plot_obs: a bool, if True, plots the observation of the GP, defaults to True.
        :param plot_CI: a bool, if True, plots the CI of the GP, with the correspondent quantile, defaults to True.
        :param plot_samples: a bool, if True, plots the samples of the GP, defaults to False.
        :param ylim_by_CI: a bool, if True, the y axis fits according to the CI, if False, the y axis fits according to
        the prediction curve, defaults to True.
        :param statistic: a string, determines the statistic ('Mean' or 'Median') with the prediction will be
        calculated, defaults to 'Mean'. the prediction will be the median of the samples, defaults to True.
        :param samples_kwargs: a dictionary, the kwargs for the plot of samples.
        :param CI_kwargs: a dictionary, the kwargs for the plot of CI.
        :param obs_kwargs: a dictionary, the kwargs for the plot of observations.
        :param real_kwargs: a dictionary, the kwargs for the plot of real signal.
        :param pred_kwargs: a dictionary, the kwargs for the plot of prediction.

        :return: None, if return_pred True, returns the DataFrame with the prediction, if return_samples False
            returns the Dataframe with the Samples, if both are True, returns both DataFrames pred and samples.
        """
        if prior:
            pred, samples = self.predict(self.dt.index, quantiles=quantile, nsamples=nsamples,
                                       samples=True, noise=noise, prior=True)
        else:
            pred, samples = self.predict(self.dt.index, quantiles=quantile, nsamples=nsamples,
                                        samples=True, noise=noise)

        x = pred[self.dt.inputs].to_numpy().squeeze()
        real = pred[self.dt.outputs].to_numpy().squeeze()
        CI = 1 - 2 * quantile
        quantile_low = int(quantile*100)
        quantile_high = int(100 - quantile*100)
        confidence_low = pred['Quantile ' + str(quantile_low)]
        confidence_high = pred['Quantile ' + str(quantile_high)]
        statistic = statistic.lower().capitalize()
        med_post = pred[statistic]

        if valid_index is None:
            valid_index = self.dt.index
        mape = round(MAPE(self, pred, valid_index, statistic), 2)
        mae = round(MAE(self, pred, valid_index, statistic), 2)

        fig = plt.figure(figsize=(20,6))

        if plot_samples == True:
            for i in range(len(samples)):
                sample = samples.iloc[i]
                plt.plot(x, sample, **samples_kwargs)

        if plot_CI == True:
            plt.fill_between(x,
                confidence_low,
                confidence_high,
                **CI_kwargs,
                label=str(100*CI) + '% CI')

        plt.plot(x, real, **real_kwargs)

        if plot_obs == True:
            obs_index = self.obs_index
            obs_x = pred[self.dt.inputs].loc[self.obs_index].sort_index()
            obs_y = pred[self.dt.outputs].loc[self.obs_index].sort_index()
            plt.scatter(obs_x, obs_y, **obs_kwargs)

        plt.plot(x, med_post, **pred_kwargs)

        leg = plt.legend(ncol=4, frameon=True, shadow=False, loc=9, edgecolor='k', fontsize = 18)
        frame = leg.get_frame()
        frame.set_facecolor('0.9')
        plt.ylabel(y_label, fontsize = 18)
        plt.xlabel(x_label, fontsize = 18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(x[0], x[len(x)-1])

        if ylim_by_CI == True:
            minim = min(confidence_low.min(), real.min())
            maxim = max(confidence_high.max(), real.max())
        else:
            minim = min(med_post.min(), real.min())
            maxim = max(med_post.max(), real.max())

        dif = (maxim - minim)*0.1
        #plt.ylim(minim - dif, maxim + dif)
        plt.title('{0} | MAPE={1:.2f}% | MAE={2:.2f}'.format(title, mape, mae), fontsize = 20)
        plt.tight_layout()
        plt.show()

        if return_pred == True and return_samples == True:
            return samples, pred
        if return_pred == True:
            return pred
        if return_samples == True:
            return samples

    def get_priors(self):
        """
        Save the value of sample in a dict

        :param npriors: an int, the number of groups
        :param ngroups: an int, the number of groups
        :return: a dict, value of sample for each prior and group
        """
        prior_dict = {}
        if list(self.priors.keys())[0] == 'prior01':
            for prior in list(self.marginals.keys()):
                for group in list(self.marginals[prior].p.keys()):
                    prior_dict[(prior, group)] = (self.marginals[prior].p[group].data.clone()
                                                  .detach().cpu().numpy())

        else:
            for prior in list(self._priors_dict.keys()):
                for group in list(self._priors_dict[prior].p.keys()):
                    prior_dict[(prior, group)] = (self._priors_dict[prior].p[group].data.clone()
                                                  .detach().cpu().numpy())
        return prior_dict
def MAPE(tgp, pred, val_index, statistic='Mean'):

    """
    Calculates the MAPE of the prediction of a TGP.

    :param tgp: a TGP object, the TGP.
    :param pred: a dataframe, contains the prediction of the TGP.
    :val_index: a numpy.ndarray, the index of values for validation.
    :param statistic: a string, determines the statistic ('Mean' or 'Median') with wich the prediction will be calculated,
            defaults to 'Mean'.
    :return: an int, the MAPE of the prediction.
    """

    statistic = statistic.lower().capitalize()
    real_valid = pred.iloc[val_index][tgp.dt.outputs].sort_index().to_numpy().squeeze()
    med_pred_valid = pred.iloc[val_index][statistic].sort_index().to_numpy().squeeze()

    APE = []
    for i in range(len(val_index)):
        if real_valid[i] == 0:
            pass
        else:
            err = abs((real_valid[i] - med_pred_valid[i]) / real_valid[i])
            APE.append(err)
    MAPE = (sum(APE)/len(APE))

    return MAPE*100

def MAE(tgp, pred, val_index, statistic='Mean'):
    """
    Calculates the MAE of the prediction of a TGP.
    :param tgp: a TGP object, the TGP.
    :param pred: a dataframe, contains the prediction of the TGP.
    :val_index: a numpy.ndarray, the index of values for validation.
    :param statistic: a string, determines the statistic ('Mean' or 'Median') with wich the prediction will be calculated,
        defaults to 'Mean'.
    :return: an int, the MAE of the prediction.
    """

    statistic = statistic.lower().capitalize()
    real_valid = pred.iloc[val_index][tgp.dt.outputs].sort_index().to_numpy().squeeze()
    med_pred_valid = pred.iloc[val_index][statistic].sort_index().to_numpy().squeeze()

    MAE = abs((real_valid - med_pred_valid)).mean()

    return MAE

