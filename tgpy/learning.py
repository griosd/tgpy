import torch
from math import log
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm.notebook import tqdm
from .random import TGP
from .tensor import _device, cholesky, zero ,to_numpy
from .kernel import SE
from .prior import TgPrior
from .modules import Constant
from scipy import stats
from matplotlib import pyplot as plt


class TgLearning:
    def __init__(self, tgp: TGP, lr=0.01, index=None, pbatch=1.0, cycle=0, pot=1):
        self.tgp = tgp
        self.optimizer = optim.Adam(self.tgp.parameters(), lr=lr)
        self.parameters_list = list(self.tgp.parameters())
        self.dlogp_clamp = 1
        self.cycle = cycle
        self.pot = pot
        self.pbatch = 1.0 if pbatch is False or pbatch is True else pbatch
        self.eg2 = zero
        self.grassman_A = None
        self.T = None
        self.index = index
        if self.index is None and self.tgp.obs_x is not None:
            self.index = torch.arange(len(self.tgp.obs_x), device=_device)
            self.index_observed = torch.clone(self.index)
        else:
            self.index_observed = None

    @property
    def nbatch(self):
        return int(self.nobs * self.pbatch)

    @property
    def nparams(self):
        return self.priors_list[0].dim

    @property
    def nobs(self):
        return len(self.index) if self.index is not None else 0

    @property
    def priors_list(self):
        return self.tgp.priors_list

    @property
    def priors_dict(self):
        return self.tgp.priors

    @property
    def index_batch(self):
        """Returns a torch.tensor with a random permutation of a subset (of size nbatch) of time indices."""
        if (self.nbatch is not None) and (self.nbatch < self.nobs):
            torch.randperm(self.nobs, out=self.index)
            return self.index_observed[self.index[:self.nbatch]].sort().values
        else:
            return Ellipsis

    @property
    def niters(self):
        return 0

    def review(self, niters: int = 100, nreview: int = 1, nprior: int = 0, ngroup: int = 0, mcmc: bool = True,
               grassman: bool = True, grad_method: str ='stochastic'):
        """
        Save a dictionary with nreview samples

        :param niters: an int, the algorithm's number of iterations.
        :param nreview: an int, number of round in which review the training
        :param nprior: an int, number of prior
        :param ngroup: an int, number of group
        :param mcmc: a bool, if it's true run mcmc in the training
        :return: a dict, value of sample for each prior and group
        """

        review_dict = {}
        self.eg2 = zero

        for k in range(int(nreview)):
            self.execute_svgd(niters=int(np.ceil(niters / nreview)), reset=False, mcmc=mcmc, grassman=grassman,
                              grad_method=grad_method, A=self.grassman_A, T=self.T)

            review_dict['r{}'.format(k)] = self.tgp.get_priors()
        return review_dict


    def plotKSCDF(self, theorical: dict, review_dict: dict, ncols: int = 2, rprior: int = 0, rgroup: int = 0,
                  title: str = 'metodo'):
        """
        Plot the CDF of both samples with their KS_statistic

        :param theorical: a dict, the theorical sample for each prior and group
        :param review_dict: a dict, contains the sample each nreview iterations
        :param ncols: an int, the plot's number of columns, defaults to 2
        :param rprior: an int, prior to review
        :param rgroup: an int, group to review
        """
        theo = theorical['prior{}'.format(rprior), 'g{}'.format(rgroup)]
        keys = list(review_dict.keys())
        nrows = int(np.ceil(len(review_dict.keys()) / ncols))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows), squeeze=False)
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Numpys de datos
        for i, key in enumerate(review_dict.keys()):
            sample = np.sort(review_dict[key]['prior{}'.format(rprior), 'g{}'.format(rgroup)])
            theo = np.sort(theo)

            data1 = theo
            data2 = sample

            # Concatenar los datos sin ordenar
            data_all = np.concatenate([data1, data2])

            # CDF: Cumulative distribution function
            # Calcular la CDF empírica normalizada para data1
            cdf1 = np.searchsorted(data1, data_all, side='right') / data1.size

            # Calcular la CDF empírica normalizada para data2
            cdf2 = np.searchsorted(data2, data_all, side='right') / data2.size

            # Calcular la estadística KS
            ks_statistic = np.abs(cdf1 - cdf2).max()
            index_ks = np.argmax(np.abs(cdf1 - cdf2))
            location_ks = data_all[index_ks]
            res = stats.kstest(data1, data2, N=len(sample))

            # Crear el gráfico
            axi = ax[i // ncols, i % ncols]
            axi.set_title('CDF Empíricas y Estadística KS ' + keys[i])
            axi.plot(np.sort(data_all), np.sort(cdf1), label='CDF sample Teorico')
            axi.plot(np.sort(data_all), np.sort(cdf2), label='CDF sample SVGD')
            axi.plot([location_ks, location_ks], [cdf2[index_ks], cdf1[index_ks]], '--')

            axi.set_ylabel('Probabilidad Acumulada')

            axi.legend()

            # Agregar el valor de la estadística KS al gráfico
            axi.text(0.05, 0.65, f'KS = {ks_statistic:.2f}', transform=axi.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            axi.text(0.05, 0.47, f'pvalue = {res[1]:.2f}', transform=axi.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))  # No se si es necesario

        # Mostrar el gráfico
        plt.show()


    def plotKSEvolution(self, theorical: dict, review_dict: dict, rprior: int = 0,
                        rgroup: int = 0, showplot: bool = True):
        """
        Plot evolution of KS_statistic

        :param theorical: a dict, the theorical sample for each prior and group
        :param review_dict: a dict, contains the sample each nreview iterations
        :param rprior: an int, prior to review
        :param rgroup: an int, group to review
        :param showplot: a bool, if it's true the plot show
        """
        theo = theorical['prior{}'.format(rprior), 'g{}'.format(rgroup)]
        ks_statistic = []
        # Numpys de datos
        for i, key in enumerate(review_dict.keys()):
            sample = np.sort(review_dict[key]['prior{}'.format(rprior), 'g{}'.format(rgroup)])
            theo = np.sort(theo)

            data1 = theo
            data2 = sample

            # Concatenar los datos sin ordenar
            data_all = np.concatenate([data1, data2])

            # CDF: Cumulative distribution function
            # Calcular la CDF empírica normalizada para data1
            cdf1 = np.searchsorted(data1, data_all, side='right') / data1.size

            # Calcular la CDF empírica normalizada para data2
            cdf2 = np.searchsorted(data2, data_all, side='right') / data2.size

            # Calcular la estadística KS
            ks_statistic.append(np.abs(cdf1 - cdf2).max())

        # Crear el gráfico
        if showplot:
            plt.figure(figsize=(15, 7))
            plt.plot(ks_statistic, 'o-', ms=8, alpha=0.7)
            plt.ylabel('KS_statistic')
            plt.xlabel('nreview')
            plt.title('KS evolution for ' + 'prior{} '.format(rprior) + 'group{}'.format(rgroup))
            plt.ylim([0, 1])
            plt.show()
        return ks_statistic

    def sbc(self, niters_sbc: int = 100, niters_sgd=100):
        """
        Returns dataframes of distribution theta0 and theta.

        :param niters_sbc: an int number of extractions of theta_0.
        :param niters_sgd: an int, number of SGD training iterations.

        """
        samples_theta_prior = []
        samples_theta_posterior = []
        for i in range(niters_sbc):
            self.tgp.sample_priors()
            samples_theta_prior.append(pd.DataFrame({
                k: pd.Series(list(self.priors_dict[k].p.values())[0][0].detach().cpu().numpy())
                for k in self.priors_dict
            }))

            self.tgp.dt.output = to_numpy(self.tgp.prior(self.tgp.dt.index)[0, :, 0]).T
            self.tgp.dt.calculate_scale(inputs=False, outputs=True)
            self.tgp.obs(self.tgp.dt.index)
            self.execute_sgd(niters_sgd)  # Train SGD

            samples_theta_posterior.append(pd.DataFrame({
                k: pd.Series(list(self.priors_dict[k].p.values())[0].detach().cpu().numpy())
                for k in self.priors_dict
            }))

        df_prior = pd.concat(samples_theta_prior, ignore_index=True, axis=0)  # Dataframe theta_0
        df_posterior = pd.concat(samples_theta_posterior, ignore_index=True, axis=0)  # Dataframe theta

        df_prior = df_prior.astype(float)

        return df_prior, df_posterior


    def execute_sgd(self, niters, update_loss=10, langevin: bool = False, langevin_add=0.05, langevin_mult=0.01):
        """
        Executes the Stochastic Gradient Descent method langevin dynamics .

        :param niters: an int, the number of iterations for the algorithm.
        :param update_loss: an int, the number of iterations to wait to update the loss value next to the progress bar.
        :param langevin: a bool, if true, smapling with langevin dynamics.
        :param langevin_add: a float, the additive langevin noise.
        :param langevin_mult: a float, the multiplicative langevin noise.
        """

        bar = tqdm(range(niters), ncols=950)
        update_loss -= 1
        end = self.niters + niters
        _ = self.tgp.logp()
        for t in bar:
            logp = self.tgp.logp(index=self.index_batch)
            logp_nan = torch.isfinite(logp)
            loss = -logp[logp_nan].nanmean()
            self.optimizer.zero_grad()
            loss.backward()
            self.tgp.clamp_grad()
            self.optimizer.step()
            if langevin:
                for p in self.parameters_list:
                    p.data = p.data*(1+torch.randn(p.shape, device=p.device)*langevin_mult) + torch.randn(p.shape, device=p.device)*langevin_add
            self.tgp.clamp()
            self.append()
            if t % update_loss == 0:
                logp = self.tgp.logp()
                logp_median = logp.median().item()
                logp_std = (logp - logp_median).abs().median().item()
                desc = 'sgd | batch: {0:.2f}% | iters: [{1}, {2}) | logp: {3:.3f} ± {4:.3f} |'
                bar.set_description_str(desc.format(100 * self.nbatch / max(1, self.nobs), self.niters, end,
                                                    logp_median, logp_std))

    def execute_svgd(self, niters, delta=.01, A=None, Tmin=1e-4, Tmax=1e6, Ttresh=None, T=None, update_loss=10,
                     grad_method='stochastic', niter_grad=None, N=None, reset: bool = True,
                     mcmc: bool = True, grassman: bool = True):
        """
        Executes the Stein Variational Gradient Descent method with MCMC and Grassman options .

        :param niters: an int, the number of iterations for the algorithm.
        :param delta: a float, step size in Euler-Maruyama discretization of the projection dynamic.
        :param A: a torch.tensor, the initial projection matrix.
        :param Tmin: a float, the minimum value taken by the annealing temperature.
        :param Tmax: a float, the maximum value taken by the annealing temperature.
        :param Ttresh: a float, the minimum difference between the particle-averaged magnitude necessary to keep the
        temperature.
        :param update_loss: an int, the number of iterations to wait to update the loss value next to the progress bar.
        :param drop_niters: an int, the number of initial additional non-appended iterations.
        :param grad_method: a string, the method used to calculate the gradient of alpha([A]). Can take the values
        stochastic, or exact.
        :param niter_grad: an int, number of iterations in the batch stochastic gradient calculation.
        :param N: an int, batch size in the batch stochastic gradient calculation.
        :param kernel: a tgpy.kernel, the kernel to be used, if is None SE will be used.
        :param reset: a bool, if true, set eg2 to zero
        """

        bar = tqdm(range(niters), ncols=950)
        update_loss -= 1
        end = self.niters + niters
        logp = self.tgp.logp()
        no_grad = torch.no_grad()
        if reset:
            self.eg2 = zero

        with no_grad:
            sigma = []
            for name, prior in self.priors_dict.items():
                if hasattr(prior, 'd'):
                    for n, dist in prior.d.items():
                        sigma.append(dist.high*0+1)
                else:
                    for n, dist in prior.d0.items():
                        sigma.append(dist.high*0+1)
                    for n, dist in prior.d1.items():
                        sigma.append(dist.high*0+1)
            sigma = torch.stack(sigma)
            epoch = int(niters * self.cycle)
        if grassman:
            if T is None:
                T = Tmin
            dn = len(sigma)
            if A is None:
                A = torch.eye(dn)
                A = A.reshape((dn, dn, -1))
            else:
                if len(A.shape) == 2:
                    A = A[None, :, :]
            dgamma0 = 0
            if Ttresh is None:
                Ttresh = 1e-4 * A.shape[0]
            A = A.to(sigma.device)

        for t in bar:
            logp = self.tgp.logp(index=self.index_batch)
            logp_nan = torch.isfinite(logp)
            loss = -logp[logp_nan].sum()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            with no_grad:
                x = torch.stack([p if p.shape[0] > 1 else p.repeat(self.nparams, p.shape[1])
                                 for p in self.parameters_list], dim=1)
                dlogp = torch.stack([p.grad.data if p.shape[0] > 1 else p.grad.data.repeat(self.nparams, p.shape[1])
                                     for p in self.parameters_list], dim=1)
                dlogp = torch.clamp(dlogp, -self.dlogp_clamp, self.dlogp_clamp)
                annealing_gsvgd = (((t % epoch) + 1) / epoch) ** self.pot if epoch > 0 else 1

                if grassman:
                    # Direction of GSVGD
                    self.eg2, d = self.svgd_direction(x, dlogp, sigma=sigma, annealing=annealing_gsvgd,
                                                      projection=A[0], eg2=self.eg2, mcmc=mcmc,
                                                      grassman=grassman)
                    for k in range(1, A.shape[0]):
                        self.eg2, aux = self.svgd_direction(x, dlogp, sigma=sigma, annealing=annealing_gsvgd,
                                                            projection=A[k], eg2=self.eg2, mcmc=mcmc,
                                                            grassman=grassman)
                        d += aux
                else:
                    # Direction of SVGD
                    self.eg2, d = self.svgd_direction(x, dlogp, sigma=sigma, annealing=annealing_gsvgd, alpha=1,
                                                      eg2=self.eg2, mcmc=mcmc, grassman=grassman)
                for j, p in enumerate([p for p in self.parameters_list]):
                    p.data -= (d.data[:, j] if p.shape[0] > 1 else d.data[:, j].mean(dim=0, keepdim=True))
                self.tgp.clamp_grad()
                # self.optimizer.step()
                self.tgp.clamp()
                if grassman:
                    # Adapt temperature
                    dgamma = self.particle_avg_magnitude(d)
                    if max(dgamma - dgamma0, dgamma0 - dgamma) < Ttresh and T < Tmax:
                        T *= 10

            if grassman:
                # update projector according to the chosen gradient
                if grad_method == 'exact':
                    if t == 0:
                        try:
                            with no_grad:
                                for k in range(A.shape[0]):
                                    A[k] = self.svgd_update_projection(params=x, dlogp=dlogp, A=A[k], sigma=sigma, T=T,
                                                                       delta=delta, grad_method=grad_method,
                                                                       niter=niter_grad, N=N)
                        except RuntimeError:
                            print(
                                'Not enought memory for exact gradient method, changing to estochastic approximation.')
                            grad_method = 'stochastic'
                            with no_grad:
                                for k in range(A.shape[0]):
                                    A[k] = self.svgd_update_projection(params=x, dlogp=dlogp, A=A[k], sigma=sigma, T=T,
                                                                       delta=delta, grad_method=grad_method,
                                                                       niter=niter_grad, N=N)
                    else:
                        with no_grad:
                            for k in range(A.shape[0]):
                                A[k] = self.svgd_update_projection(params=x, dlogp=dlogp, A=A[k], sigma=sigma, T=T,
                                                                   delta=delta, grad_method=grad_method,
                                                                   niter=niter_grad, N=N)
                elif grad_method == 'stochastic':
                    with no_grad:
                        for k in range(A.shape[0]):
                            A[k] = self.svgd_update_projection(params=x, dlogp=dlogp, A=A[k], sigma=sigma, T=T,
                                                               delta=delta, grad_method=grad_method,
                                                               niter=niter_grad, N=N)
                else:
                    raise ValueError(
                        'grad_method must be in [stochastic/exact/approx]. grad_method selected {}'.format(grad_method))

                self.optimizer.__init__(self.parameters_list, lr=self.optimizer.param_groups[0]['lr'])
            self.append()
            if t % update_loss == 0:
                self.optimizer.__init__(self.parameters_list, lr=self.optimizer.param_groups[0]['lr'])
                logp = self.tgp.logp()
                logp_median = logp.median().item()
                logp_std = (logp - logp_median).abs().median().item()
                desc = 'psvgd | batch: {0:.2f}% | iters: [{1}, {2}) | logp: {3:.3f} ± {4:.3f} |'
                bar.set_description_str(desc.format(100 * self.nbatch / max(1, self.nobs), self.niters, end,
                                                    logp_median, logp_std))


    @staticmethod
    def svgd_direction(params: torch.Tensor, dlogp: torch.Tensor, sigma: torch.Tensor,
                       annealing: torch.Tensor, projection: torch.Tensor = None,
                       alpha: float = 1, eg2: torch.Tensor = 0, beta: float = 0.9, mcmc: bool = True,
                       grassman: bool = True):
        """
        Calculate the kernel Stein direction.

        :param params: a torch.tensor, the parameters of models.
        :param dlogp: a torch.tensor, the derivative of logp w.r. the parameters of models.
        :param sigma: a torch.tensor, the sigma of Gaussian kernel.
        :param annealing: a float, the annealing ponderator.

        :return: a torch.tensor, the kernel Stein direction-
        """
        n_samples = params.shape[0]
        if grassman:
            # Kernel calculate
            xA1 = (params.mm(projection))[None, :, :]
            xA2 = (params.mm(projection))[:, None, :]
            d = (xA2 - xA1) / (sigma @ projection)
            dists = d.pow(2).sum(axis=-1)
            h = 0.5 * dists.median().item() / log(n_samples + 1)
            h = h if h > 1e-3 else 1e-3
            k = torch.exp(- dists[:, :, None] / (2 * h)).squeeze()
            k_der = (-2 * k[:, :, None] * (xA1 - xA2)[None, None, :, :, :] /
                     ((2 * h) ** 0.5 * sigma @ projection).pow(2)).squeeze().reshape(n_samples, n_samples, -1)

            # Update rule
            ks = dlogp.matmul(projection.mm(projection.transpose(0, 1)).transpose(0, 1))
            ks = (ks[:, None, :] * k[:, :, None]).sum(axis=0)
            kd = k_der.matmul(projection.transpose(0, 1)).sum(axis=0)
        else:
            # Kernel calculate
            d = (params[:, None, :] - params[None, :, :]) / sigma
            dists = d.pow(2).sum(axis=-1)
            h = 0.5 * dists.median().item() / log(n_samples + 1)
            h = h if h > 1e-3 else 1e-3
            k = torch.exp(- dists / (2 * h))
            k_der = d * k[:, :, None] / (sigma * h)

            # update rule
            ks = k.mm(dlogp)
            kd = k_der.sum(axis=0)

        df = (annealing * ks + kd) / n_samples
        if mcmc:
            L = cholesky(k)
            eg2 = beta * eg2 + (1 - beta) * (df ** 2).sum()
            alpha /= torch.sqrt(eg2)
            return (eg2, alpha * df + ((2 * alpha / n_samples) ** 0.5) *
                    L.mm(torch.randn(dlogp.shape, device=L.device)))

        else:
            return None, df
    # Grassman #
    @staticmethod
    def particle_avg_magnitude(direction):
        """
        Calculates de particle-averaged magnitude.

        :param direction: a torch.tensor, the direction.

        :return: a float, the particle-averaged magnitude in the direction d.
        """
        norm = direction.abs().max(axis=1).values
        return norm.mean()


    @staticmethod
    def alpha_A(params: torch.Tensor, dlogp: torch.Tensor, A: torch.Tensor, kernel):
        """
        Calculate the kernel stein discrepancy induced by the matrix A.

        :param params: a torch.tensor, the parameters of models.
        :param dlogp: a torch.tensor, the derivative of logp w.r. the parameters of the model.
        :param A: a torch.tensor, the projection matrix.
        :param kernel: a tgpy.kernel, the kernel to be used.

        :return: a float, the kernel stein discrepancy induced by the matrix A, valuated in the empirical distribution of the data.
        """
        A = A.requires_grad_()
        dlog1 = dlogp.mm(A)[:, None, :]
        dlog2 = dlogp.mm(A)[None, :, :]
        X1 = params.clone().detach().requires_grad_()
        X2 = params.clone().detach().requires_grad_()
        n_samples = params.shape[0]

        k = kernel.forward((X1.mm(A))[None, :, :], (X2.mm(A))[:, None, :]).squeeze()
        k_der = kernel.grad((X1.mm(A))[None, :, :], (X2.mm(A))[:, None, :]).squeeze().reshape(n_samples, n_samples, -1)
        hess_trace = kernel.hess_trace((X1.mm(A))[None, :, :], (X2.mm(A))[:, None, :])

        KSDA1 = (dlog1 * dlog2).sum(axis=-1) * k
        KSDA1 = KSDA1.mean()
        KSDA2 = -(dlog1 * k_der).sum(axis=-1)
        KSDA2 = 2 * KSDA2.mean()
        KSDA3 = hess_trace.mean()

        return KSDA1 + KSDA2 + KSDA3

    @staticmethod
    def projection_A(A: torch.Tensor):
        """
        Calculate the projection operator Id - AA^T.

        :param A: a torch.tensor, the projection matrix.
        :return: a torch.tensor, the projection operator Id - AA^T.
        """
        d = A.shape[0]
        AAt = A.mm(torch.transpose(A, 0, 1))
        return torch.eye(d, device=_device) - AAt

    @staticmethod
    def gradient_projection(delta, A):
        """
        Calculate an approximation of the exponential map using a thin singular value descomposition.

        :param delta: a torch.tensor, ehere the function will be evaluated.
        :param A: a torch.tensor, the projection matrix.

        :return: a torch.tensor, the approximation of the exponential map evaluated in delta.
        """
        U, S, V = torch.svd(A + delta, compute_uv=True)
        return U.mm(V.transpose(0, 1))


    def svgd_update_projection(self, params: torch.Tensor, dlogp: torch.Tensor, A: torch.Tensor, sigma: torch.Tensor,
                               T, delta, grad_method='stochastic', niter=None, N=None):
        """
        Calculate the projection that minimize logp.

        :param params: a torch.tensor, parameters of the model.
        :param dlogp: a torch.tensor, the derivative of logp w.r. the parameters of models.
        :param A: a torch.tensor, the initial projection matrix.
        :param sigma: a torch.tensor, the relevance parameter of kernel.
        :param T: a float, the temperature.
        :param delta: a float, step size in Euler-Maruyama discretization of the projection dynamic.
        :param grad_method: a string, the method used to calculate the gradient of alpha([A]). Can take the values
        stochastic, or exact.
        :param niter: an int, number of iterations in the batch stochastic gradient calculation.
        :param N: an int, batch size in the batch stochastic gradient calculation.

        :return: a torch.tensor, the matrix projection that minimizes the loss. If grad_method is exact, the real value
        of grad(alpha([A])) is used and if grad_method is stochastic, the gradient is approximated stochastically.
        """
        # Set N and niter for stochastic method
        if N is None:
            N = int(0.05 * params.shape[0])
            N = min(max(N, 1), 500)
        if niter is None:
            niter = 1

        if grad_method == 'stochastic':
            dalpha = self.stochastic_grad_alpha(A=A, params=params, dlogp=dlogp, sigma=sigma, niter=niter, N=N)
        elif grad_method == 'exact':
            dalpha = self.grad_alpha(A=A, x=params, dlogp=dlogp, sigma=sigma)
        else:
            raise ValueError(
                'grad_method must be in [stochastic/exact]. grad_method selected {}'.format(grad_method))

        # Calculate delta for singular value descomposition.
        zhi = torch.normal(mean=torch.zeros_like(A), std=torch.ones_like(A))
        Delta = delta * self.projection_A(A).mm(dalpha)
        Delta += (2 * delta * T) ** 0.5 * self.projection_A(A).mm(zhi)

        return self.gradient_projection(Delta, A)

    @staticmethod
    def grad_alpha(A: torch.Tensor, x: torch.Tensor, dlogp: torch.Tensor, sigma: torch.Tensor):
        """
        Calculate the gradient of alpha([A]).

        :param A: a torch.tensor, the projection matrix.
        :param x: a torch.tensor, parameters of the model.
        :param dlogp: a torch.tensor, the derivative of logp w.r. the parameters of models.
        :param sigma: a torch.tensor, the sigma of Gaussian kernel.
        :return: a torch.tensor, the exact value of the gradient of alpha([A]) evaluated at A.
        """
        n_samples = x.shape[0]
        # Projection of params in A
        xA1 = (x.mm(A))[None, :, :]
        xA2 = (x.mm(A))[:, None, :]

        # Kernel operator
        d = (xA2 - xA1) / (sigma @ A)
        dists = d.pow(2).sum(axis=-1)
        h = 0.5 * dists.median().item() / log(n_samples + 1)
        h = h if h > 1e-3 else 1e-3
        k = torch.exp(- dists[:, :, None] / (2 * h)).squeeze()

        delta_pq = (dlogp[:, None, :] - dlogp[None, :, :])

        # Calculate of gradient
        term1 = ((k * (delta_pq[:, :, :, None] * delta_pq[:, :, None, :]).sum(axis=-1).mean(axis=-1))[:, :, None, None]
                 * A[None, None, :, :])
        term1 = term1.mean(axis=(0, 1))

        term2 = (((delta_pq[:, :, :, None] * A[None, None, :, :]).sum(axis=-2))[:, :, :, None] *
                 ((delta_pq[:, :, :,None] * A[None, None, :, :]).sum(axis=-2))[:, :, None, :])
        term2 = term2.sum(axis=-1).mean(axis=-1)
        term2 = ((k * (x[None, :, :] - x[:, None, :]).pow(2).sum(axis=-1) * term2)[:, :, None, None] *
                 A[None, None, :, :])
        term2 = term2.mean(axis=(0, 1))

        return 2 * (term1 + term2)

    def stochastic_grad_alpha(self, A: torch.Tensor, params: torch.Tensor, dlogp: torch.Tensor, sigma: torch.Tensor,
                              niter: int = 30, N: int = 20):

        """
        Approximate the gradient of alpha([A]) stochastically.

        :param A: a torch.tensor, the projection matrix.
        :param params: a torch.tensor, parameters of the model.
        :param dlogp: a torch.tensor, the derivative of logp w.r. the parameters of models.
        :param sigma: a torch.tensor, the sigma of Gaussian kernel.
        :param niter: an int, number of iterations in the stochastical approximation.
        :param N: an int, size of the subsample for the stochastical approximation.

        :return: a torch.tensor, the approximation of the gradient of alpha([A]) evaluated at A.
        """
        grad = torch.zeros_like(A)
        for t in range(niter):
            # Stochastic sample of params and logp
            sample_idx = np.random.choice(params.shape[0], N)
            x_t = params[sample_idx, :]
            dlogp_t = dlogp[sample_idx, :]

            # Projection of params in A
            n_samples = params.shape[0]
            xA1 = (x_t.mm(A))[None, :, :]
            xA2 = (x_t.mm(A))[:, None, :]

            # Kernel operator
            d = (xA2 - xA1) / (sigma @ A)
            dists = d.pow(2).sum(axis=-1)
            h = 0.5 * dists.median().item() / log(n_samples + 1)
            h = h if h > 1e-3 else 1e-3
            k = torch.exp(- dists[:, :, None] / (2 * h)).squeeze()

            delta_pq = (dlogp_t[:, None, :] - dlogp_t[None, :, :])

            # Calculate of gradient
            term1 = ((k * (delta_pq[:, :, :, None] * delta_pq[:, :, None, :]).sum(axis=-1).
                      mean(axis=-1))[:, :, None, None] * A[None, None, :, :])
            term1 = term1.mean(axis=(0, 1))

            term2 = (((delta_pq[:, :, :, None] * A[None, None, :, :]).sum(axis=-2))[:, :, :, None] *
                     ((delta_pq[:, :, :, None] * A[None, None, :, :]).sum(axis=-2))[:, :, None, :])
            term2 = term2.sum(axis=-1).mean(axis=-1)
            term2 = ((k * (x_t[None, :, :] - x_t[:, None, :]).pow(2).sum(axis=-1) * term2)[:, :, None, None] *
                     A[None, None, :, :])
            term2 = term2.mean(axis=(0, 1))

            grad += 2 * (term1 + term2) / niter

        return grad


    def append(self):
        pass
