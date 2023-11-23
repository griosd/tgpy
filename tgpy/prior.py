import torch
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.distributions as dist
import torch.distributions.constraints as const
import torch.distributions.transforms as trans
from .tensor import zero, one, two, eps4, to_numpy
from .utils import DictObj, color_next
from scipy.stats import gaussian_kde


class TgPrior(nn.Module):
    def __init__(self, name, dim=1):
        super(TgPrior, self).__init__()
        self.name = name
        self.dim = dim
        self.color = None

class TgPriorUnivariate(TgPrior):
    def __init__(self, name, parameters, dim=1, low=zero, high=one, alpha=None, beta=None, mean=None, mode=None):
        super(TgPriorUnivariate, self).__init__(name, dim=dim)
        self.parameters = parameters
        self.d = DictObj({p: BetaLocation(low=low, high=high, alpha=alpha, beta=beta, mean=mean, mode=mode)
                          for p in parameters})
        self.p = nn.ParameterDict({p: nn.Parameter(self.d[p].sample((self.dim,))) for p in parameters})

    def forward(self, *args, **kwargs):
        return torch.stack(tuple(self.p.values()), dim=1)

    def logp(self):
        return sum(self.d[p].log_prob(self.p[p]) for p in self.parameters)

    def clamp(self, tol=1e-6):
        for n in self.parameters:
            clamp_nan(self.p[n].data, self.d[n].low+tol, self.d[n].high-tol, nan=True)

    def clamp_grad(self, tol=1e-6):
        for n in self.parameters:
            t = self.p[n].grad.data
            t.masked_fill_(t.ne(t), 0)

    def sample_params(self):
        for n in self.parameters:
            self.p[n].data = self.d[n].sample((self.dim,))

    def plot(self, nspace=100, ncols=2, mean=False, median=False, samples=True, kde=True, color=None, alpha=0.5,
             save_as=None, *args, **kwargs):
        """
        Plots the priors of each group of an NgPrior.

        :param nspace: an int, the steps of the x-axis.
        :param ncols: an int, the number of columns for the plot.
        :param mean: a boolean, whether to plot the mean of each prior as a vertical line.
        :param median: a boolean, whether to plot the median of each prior as a vertical line.
        :param samples: a boolean, whether to scatter-plot samples.
        :param kde: a boolean, whether to plot the kde of samples.
        :param color: a string, a color.
        :param alpha: a float, a value used for blending.
        :param save_as: a string, if is not None, the plot will be saved in the directory indicated in the parameter.
        :param args: arguments to pass to BetaLocation.plot.
        :param kwargs: keyword arguments to pass to BetaLocation.plot.
        """
        if color is not None:
            self.color = color
        elif self.color is None:
            self.color = color_next()

        nrows = int(np.ceil(len(self.d) / ncols))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows), squeeze=False)
        for i, (g, d) in enumerate(self.d.items()):
            axi = ax[i // ncols, i % ncols]
            axi.set_title(self.name + ' - ' + g)
            # noinspection PyArgumentList
            ylim = d.plot(nspace=nspace, ax=axi, mean=mean, median=median, alpha=alpha, color=self.color,
                          return_ylim=True, *args, **kwargs)
            if samples:
                s = to_numpy(self.p[g].data)
                sy = to_numpy(d.log_prob(self.p[g].data).squeeze().exp())
                sy[(sy != sy) | (sy > ylim)] = ylim
                if s.size == 1:
                    axi.plot([s.item(), s.item()], [0, sy * 0.98], lw=4.0, color=self.color, alpha=1.0)
                else:
                    axi.scatter(s, np.random.rand(len(s)) * sy, color=self.color, alpha=0.5)
                    if kde and s.size > 1 and np.unique(s).size > 1:
                        sb.kdeplot(s, ax=axi, color=self.color, alpha=0.5, fill=True, ls='--')

        if save_as is not None:
            fig.savefig(save_as, bbox_inches='tight', dpi=300)
        #plt.show()
        return ax
    def l2error(self, nspace=100, ncols=2, color=None, plot=True):
        """
        Calculate the mean squared error between the prior distribution and the empirical kde distribution
        for each group of an NgPrior.

        :param nspace: an int, the steps of the x-axis.
        :param ncols: an int, the number of columns for the plot.
        :param color: a string, color of the plot.
        :param plot: a boolean, wheter to plot the distributions.

        :returns: a torch.tensor with the mean squared error of each dimension.
        """
        MSES = torch.zeros(len(self.d.items()))

        if color is not None:
            self.color = color
        elif self.color is None:
            self.color = color_next()

        nrows = int(np.ceil(len(self.d) / ncols))
        if plot:
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows), squeeze=False)
        for i, (g, d) in enumerate(self.d.items()):
            
            s = to_numpy(self.p[g].data)
            xmin = min(min(s), d.low.item())
            xmin = min(0.9 * xmin, 1.1 * xmin)
            xmax = max(max(s), d.high.item())
            xmax = max(0.9 * xmax, 1.1 * xmax)
            x = np.linspace(xmin, xmax, nspace)
            kde = gaussian_kde(s)
            density = to_numpy(d.log_prob(torch.Tensor(x).to(self.p[g].data.device)))
            density[np.isnan(density)] = 0.0
            density_max = density[np.isfinite(density)].max()
            density[~np.isfinite(density)] = density_max
            sy = np.exp(density.squeeze())
            sy[density == 0.0] = 0.0
            mean_error = ((kde(x) - sy) ** 2).mean()
            MSES[i] = mean_error

            if plot:
                axi = ax[i // ncols, i % ncols]
                axi.plot(x, kde(x), color=self.color, linestyle='--')
                axi.plot(x, sy, color=self.color)
                axi.fill_between(x, sy, kde(x), color=self.color, alpha=0.5)
                axi.set_title(self.name + ' - ' + g + '/ MSE: {:.2e}'.format(mean_error))
        
        if plot:
            plt.show()

        return MSES

class TgPriorMarginal(nn.Module):
    def __init__(self, name, priors, marginal=0):
        super(TgPriorMarginal, self).__init__()
        self.name = name
        self.color = None
        self.priors = priors
        self.marginal = marginal

    def forward(self, *args, **kwargs):
        if self.marginal == 0:
            return torch.stack(tuple(self.priors.p0.values()), dim=1)
        else:
            return torch.stack(tuple(self.priors.p1.values()), dim=1)

    @property
    def p(self):
        if self.marginal == 0:
            return self.priors.p0
        else:
            return self.priors.p1

    def plot(self, *args, **kwargs):
        self.priors.plot_marginal(self.marginal, self.name, *args, **kwargs)

class TgPriorBivariate(TgPrior):
    def __init__(self, name, priors, r, dim=1):
        super(TgPriorBivariate, self).__init__(name, dim=dim)
        self.d0 = priors[0].d
        self.d1 = priors[1].d
        self.parameters = priors[0].parameters
        self.r = r
        x0 = dict()
        x1 = dict()
        for p in self.parameters:
            x0[p] = self.d0[p].sample((self.dim,))
            x1[p] = self.d1[p].sample((self.dim,))
        self.p0 = nn.ParameterDict({p: nn.Parameter(x0[p] + self.r * x1[p]) for p in self.parameters})
        self.p1 = nn.ParameterDict({p: nn.Parameter(x1[p]) for p in self.parameters})

    def marginal(self, name, marginal=0):
        return TgPriorMarginal(name, self, marginal=marginal)

    def forward(self, *args, **kwargs):
        return torch.stack(tuple(self.p0.values()), dim=1), torch.stack(tuple(self.p1.values()), dim=1)

    def logp(self):
        return sum(self.d0[p].log_prob(self.p0[p] - self.r * self.p1[p]) for p in self.parameters) + sum(self.d1[p].log_prob(self.p1[p]) for p in self.parameters)

    def clamp(self, tol=1e-6):
        for n in self.parameters:
            clamp_nan(self.p1[n].data, self.d1[n].low + tol, self.d1[n].high - tol, nan=True)
            clamp_nan2(self.p0[n].data, self.d0[n].low + self.p1[n].data * self.r + tol, self.d0[n].high
                       + self.p1[n].data * self.r - tol, nan=True)
            # clamp_nan2(self.p0[n].data, self.d0[n].low, self.d0[n].high)

    def clamp_grad(self, tol=1e-6):
        for n in self.parameters:
            t = self.p0[n].grad.data
            t.masked_fill_(t.ne(t), 0)
            t = self.p1[n].grad.data
            t.masked_fill_(t.ne(t), 0)

    def sample_params(self):
        for n in self.parameters:
            x0 = self.d0[n].sample((self.dim,))
            x1 = self.d1[n].sample((self.dim,))
            self.p0[n].data = x0 + self.r * x1
            self.p1[n].data = x1

    def plot(self):
        ncols = 3
        nrows = 2
        for g in self.p0.keys():
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows), squeeze=False)
            x0 = to_numpy(self.p0[g].data - self.r * self.p1[g].data)
            x1 = to_numpy(self.p1[g].data)

            y0 = to_numpy(self.p0[g].data)
            y1 = to_numpy(self.p1[g].data)

            ax[0,0].set_title('prior0 - ' + g + ' - ' + 'x0')
            sb.kdeplot(x0, ax=ax[0,0])
            ax[0,1].set_title('prior1 - ' + g + ' - ' + 'x1')
            sb.kdeplot(x1, ax=ax[0,1])
            ax[0,2].set_title('prior01 - ' + g + ' - ' + '(x0,x1)')
            ax[0,2].scatter(x0, x1, alpha=0.5)

            ax[1,0].set_title('prior0 - ' + g + ' - ' + 'y0')
            sb.kdeplot(y0, ax=ax[1, 0])
            ax[1,1].set_title('prior1 - ' + g + ' - ' + 'y1')
            sb.kdeplot(y1, ax=ax[1, 1])
            ax[1,2].set_title('prior01 - ' + g + ' - ' + '(y0,y1)')
            ax[1,2].scatter(y0, y1, alpha=0.5)
            plt.show()

    def plot_marginal(self, marginal=0, name=None, nspace = 100, ncols = 2, mean = False, median = False, samples = True, kde = True, color = None, alpha = 0.5,
        save_as = None, *args, ** kwargs):
        """
        Plots the priors of each group of an NgPrior.

        :param nspace: an int, the steps of the x-axis.
        :param ncols: an int, the number of columns for the plot.
        :param mean: a boolean, whether to plot the mean of each prior as a vertical line.
        :param median: a boolean, whether to plot the median of each prior as a vertical line.
        :param samples: a boolean, whether to scatter-plot samples.
        :param kde: a boolean, whether to plot the kde of samples.
        :param color: a string, a color.
        :param alpha: a float, a value used for blending.
        :param save_as: a string, if is not None, the plot will be saved in the directory indicated in the parameter.
        :param args: arguments to pass to BetaLocation.plot.
        :param kwargs: keyword arguments to pass to BetaLocation.plot.
        """
        if color is not None:
            self.color = color
        elif self.color is None:
            self.color = color_next()

        if marginal == 0:
            d = self.d0
            p = self.p0
        else:
            d = self.d1
            p = self.p1

        nrows = int(np.ceil(len(d) / ncols))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows), squeeze=False)
        for i, (g, d) in enumerate(d.items()):
            axi = ax[i // ncols, i % ncols]
        axi.set_title(name + ' - ' + g)
        # noinspection PyArgumentList
        #ylim = d.plot(nspace=nspace, ax=axi, mean=mean, median=median, alpha=alpha, color=self.color,
        #return_ylim = True, *args, ** kwargs)
        if samples:
            s = to_numpy(p[g].data)
        #sy = to_numpy(d.log_prob(p[g].data).squeeze().exp())
        #sy[(sy != sy) | (sy > ylim)] = ylim
        #if s.size == 1:
        #    axi.plot([s.item(), s.item()], [0, sy * 0.98], lw=4.0, color=self.color, alpha=1.0)
        #else:
        #    axi.scatter(s, np.random.rand(len(s)) * sy, color=self.color, alpha=0.5)
        if True and s.size > 1 and np.unique(s).size > 1:
            sb.kdeplot(s, ax=axi, color=self.color, alpha=0.5, fill=True, ls='--')

        if save_as is not None:
            fig.savefig(save_as, bbox_inches='tight', dpi=300)
        plt.show()

@torch.jit.script
def clamp_nan(t, lower: float, upper: float, nan: bool = False):
    """
    Calls the torch.clamp_ function (inplace clamp) with more cases.

    :param t: a torch.Tensor, the tensor to be clamped.
    :param lower: a float, the lower bound.
    :param upper: a float, the upper bound.
    :param nan: a boolean, whether to fill NaN values with (lower + upper)*0.5 or not.

    :return: a clamped torch.Tensor.
    """
    middle = (lower + upper) / 2
    if nan:
        t.masked_fill_(t.ne(t), middle)
    if lower < upper:
        t.clamp_(lower, upper)
    else:
        t.clamp_(middle, middle)

@torch.jit.script
def clamp_nan2(t, lower, upper, nan: bool = False):
    """
    Calls the torch.clamp_ function (inplace clamp) with more cases.

    :param t: a torch.Tensor, the tensor to be clamped.
    :param lower: a float, the lower bound.
    :param upper: a float, the upper bound.
    :param nan: a boolean, whether to fill NaN values with (lower + upper)*0.5 or not.

    :return: a clamped torch.Tensor.
    """
    t.clamp_(lower, upper)

class BetaLocation(dist.TransformedDistribution):
    """A class used to represent the Beta Distribution used for priors."""
    arg_constraints = {'low': const.dependent, 'high': const.dependent}
    has_rsample = True

    def __init__(self, low=None, high=None, alpha=None, beta=None, mean=None, mode=None, validate_args=None):
        """
        Four out of five parameters (barrings validate_args) are needed to define the Beta Location.

        :param low: a float, the distribution's lower bound.
        :param high: a float, the distribution's upper bound.
        :param alpha: a float and shape parameter.
        :param beta: a float and shape parameter.
        :param mean: a float, the mean of the distribution.
        :param mode: a float, the mode of the distribution.
        :param validate_args: a boolean to validate the arguments.
        """
        if low is None:
            low = 0 * one
        else:
            low = low * one
        if high is None:
            high = low + 1
        else:
            high = high * one
        self.low = low * one
        self.high = high * one
        self.scale = high - low

        if alpha is None or beta is None:
            if mode is not None:
                z = torch.div(mode - self.low, self.scale)
                if alpha is None and beta is None:
                    beta = 3 + (self.low * 0)
                    alpha = torch.div(1 - torch.mul(two - beta, z), one - z)
                elif alpha is None and beta is not None:
                    alpha = torch.div(one - torch.mul(two - beta, z), one - z)
                elif beta is None and alpha is not None:
                    beta = torch.div(torch.mul(one - z, alpha) - one + torch.mul(two, z), z)
            elif mean is not None:
                z = torch.div(mean - self.low, self.scale)
                if alpha is None and beta is None:
                    alpha = 3
                    beta = 3
                elif alpha is None and beta is not None:
                    alpha = torch.div(beta * z, one - z)
                elif beta is None and alpha is not None:
                    beta = torch.div(alpha, z) - alpha
        if alpha is None:
            alpha = 3 + (low * 0)
        if beta is None:
            beta = alpha
        alpha = alpha * one
        beta = beta * one
        super(BetaLocation, self).__init__(dist.Beta(alpha, beta), trans.AffineTransform(loc=self.low,
                                                                                         scale=self.scale),
                                           validate_args=validate_args)
        self.concentration_1 = torch.stack([alpha.sub(one), beta.sub(one)], -1)
        c = self.concentration_1.add(one)
        self.norm_const = torch.lgamma(c.sum(-1)) - torch.lgamma(c).sum(-1) - self.scale.abs().log()

    @const.dependent_property
    def support(self):
        """Returns the support of the distribution."""
        return const.interval(self.low, self.high)

    def entropy(self):
        """The entropy of BetaLocation"""
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        """Dummy function of abstract Distribution."""
        raise NotImplementedError

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        :param batch_shape: list of ints,  the desired expanded size.
        :param _instance: a BetaLocation object ,new instance provided by subclasses that need to override `.expand`.

        :return:a new BetaLocation instance with batch dimensions expanded to `batch_size`.
        """
        new = self._get_checked_instance(BetaLocation, _instance)
        return super(BetaLocation, self).expand(batch_shape, _instance=new)

    def log_prob_nojit(self, value):
        """
        Evaluates the log distribution.

        :param value: a torch.tensor, the point to evaluate.

        :return: a torch.tensor, the value of the evaluation.
        """
        value_inv = value.sub(self.low).div(self.scale)
        return (torch.log(torch.stack([value_inv, one.sub(value_inv)],
                                      -1)).mul(self.concentration_1)).sum(-1).add(self.norm_const)

    def log_prob(self, value: torch.Tensor):
        """
        Evaluates the log distribution wrapping a jittered function.

        :param value: a torch.tensor, the point to evaluate.

        :return: a torch.tensor, the value of the evaluation.
        """
        r = torch.zeros(value.size(), device=value.device)
        logp_beta_location(r, value, self.low, self.scale, self.concentration_1, self.norm_const, one)
        return r

    @property
    def alpha(self):
        """Alias of the concentration parameter 1."""
        return self.base_dist.concentration1

    @property
    def beta(self):
        """Alias of the concentration parameter 0."""
        return self.base_dist.concentration0

    @property
    def mean(self):
        """Alias of the mean."""
        return self.low + self.base_dist.mean * self.scale

    @property
    def variance(self):
        """Alias of the variance."""
        return self.base_dist.variance * (self.scale**2)

    @property
    def mode(self):
        """Alias of the mode."""
        return self.low + self.base_dist_mode * self.scale

    @property
    def base_dist_mode(self):
        """Return the mode of the Beta location at [0, 1]."""
        concentration0 = self.base_dist.concentration0
        concentration1 = self.base_dist.concentration1
        if concentration0 > 1 and concentration1 > 1:
            return (concentration1 - 1) / (concentration1 + concentration0 - 2)
        elif concentration0 <= 1 and concentration1 <= 1:
            return concentration0 < concentration1
        elif concentration0 >= 1:
            return 0.0
        else:
            return 1.0

    def plot(self, nspace=100, ax=None, mean=True, median=True, name=None, return_ylim=False, fill=True,
             *args, **kwargs):
        """
        Plots the Beta Location distribution.

        :param nspace: an int, the steps of the x-axis.
        :param ax: a matplotlib.axes.Axes object.
        :param mean: a boolean, whether to plot the mean as a vertical line.
        :param median: a boolean, whether to plot the median as a vertical line.
        :param name: a string to annotate in the plot.
        :param return_ylim: a boolean, whether to return the 98th percentile of the Beta Location.
        :param args: arguments to pass to ax.fill_between.
        :param kwargs: keyword arguments to pass to ax.fill_between.

        :return: a plot of the Beta Location, and ylim if return_ylim = True.
        """
        ax = ax if ax else plt
        space = torch.linspace(self.low + eps4, self.high - eps4, nspace, device=self.norm_const.device)
        density = to_numpy(self.log_prob(space).exp())
        density[np.isnan(density)] = 0.0
        density_max = density[np.isfinite(density)].max()
        density[~np.isfinite(density)] = density_max
        space = to_numpy(space)
        if fill:
            ax.fill_between(to_numpy(space), 0, density, *args, **kwargs)
        else:
            ax.plot(to_numpy(space), density, *args, **kwargs)
        ylim = np.percentile(density, 98)
        density_max = density.max()
        ylim = density_max * 1.02 if density_max / ylim < 1.5 else ylim * 1.02
        if mean:
            ax.axvline(self.mean.item(), color='k', ls='--')
        if median:
            dx = (space[1] - space[0]) / 2
            distribution = (density[1:] + density[:-1]).cumsum() * dx
            ax.axvline(space[1:][distribution >= 0.5 * distribution.max()].min().item() - dx, color='r', ls=':')
        if name:
            ax.annotate(name, xy=(self.low.item(), ylim * 0.85))
        dx_lim = self.scale.item() * 5e-3
        if ax is plt:
            ax.xlim([self.low.item() - dx_lim, self.high.item() + dx_lim])
            ax.ylim([0, ylim * 1.1])
        else:
            ax.set_xlim([self.low.item() - dx_lim, self.high.item() + dx_lim])
            ax.set_ylim([0, ylim * 1.1])
        if return_ylim:
            return ylim


@torch.jit.script
def logp_beta_location(r: torch.Tensor, value: torch.Tensor, low, scale, concentration_1, norm_const, t_one):
    """
    Modifies inplace the log distribution of the beta location.

    :param r: a torch.tensor, the point to evaluate.
    :param value: a torch.tensor, a distribution parameter.
    :param low:a torch.tensor, a distribution parameter.
    :param scale:a torch.tensor, a distribution parameter.
    :param concentration_1:a torch.tensor, a distribution parameter.
    :param norm_const:a torch.tensor, a distribution parameter.
    :param t_one:a torch.tensor with constant one.
    """
    value_inv = value.sub(low).div(scale)
    r.add_((torch.log(torch.stack([value_inv, t_one.sub(value_inv)], -1)).mul(concentration_1)).sum(-1).add(norm_const))


@torch.jit.script
def logp_beta_location_groups(r: torch.Tensor, value: torch.Tensor, low, scale, concentration_1, norm_const, t_one):
    """
    Modifies inplace the log distribution of the beta location for all groups.

    :param r: a torch.tensor, the point to evaluate.
    :param value: a torch.tensor, a distribution parameter.
    :param low:a torch.tensor, a distribution parameter.
    :param scale:a torch.tensor, a distribution parameter.
    :param concentration_1:a torch.tensor, a distribution parameter.
    :param norm_const:a torch.tensor, a distribution parameter.
    :param t_one:a torch.tensor with constant one.
    """
    value_inv = value.sub(low).div(scale)
    r.add_((torch.log(torch.stack([value_inv, t_one.sub(value_inv)], -1)).mul(concentration_1)).sum(-1).add(
        norm_const).sum((-1, -2)))


