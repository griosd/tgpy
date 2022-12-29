import torch
from math import log
import torch.optim as optim
from tqdm.notebook import tqdm
from .random import TGP
from .tensor import _device


class TgLearning:

    def __init__(self, tgp: TGP, lr=0.01, index=None, pbatch=1.0, cycle=0, pot=1, rand_pert=0):
        self.tgp = tgp
        self.optimizer = optim.Adam(self.tgp.parameters(), lr=lr)
        self.parameters_list = list(self.tgp.parameters())
        self.dlogp_clamp = 10
        self.cycle = cycle
        self.pot = pot
        self.rand_pert = rand_pert
        self.pbatch = 1.0 if pbatch is False or pbatch is True else pbatch

        self.index = index
        if self.index is None:
            self.index = torch.arange(len(self.tgp.obs_x), device=_device)
        self.index_observed = torch.clone(self.index)

    @property
    def nbatch(self):
        return int(self.nobs * self.pbatch)

    @property
    def nparams(self):
        return self.priors_list[0].dim

    @property
    def nobs(self):
        return len(self.index)

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

    def execute_sgd(self, niters, update_loss=10, drop_niters=10):
        bar = tqdm(range(niters), ncols=950)
        update_loss -= 1
        end = self.niters + niters
        _ = self.tgp.logp()
        no_grad = torch.no_grad()
        with no_grad:
            sigma = []
            for name, prior in self.priors_dict.items():
                for n, dist in prior.d.items():
                    sigma.append(dist.high - dist.low)
            sigma = torch.stack(sigma)
            epoch = int(niters * self.cycle)
        for t in range(drop_niters):
            logp = self.tgp.logp(index=self.index_batch)
            logp_nan = torch.isfinite(logp)
            loss = -logp[logp_nan].nanmean()
            self.optimizer.zero_grad()
            loss.backward()
            self.tgp.clamp_grad()
            self.optimizer.step()
            self.tgp.clamp()

        for t in bar:
            logp = self.tgp.logp(index=self.index_batch)
            logp_nan = torch.isfinite(logp)
            loss = -logp[logp_nan].nanmean()
            self.optimizer.zero_grad()
            loss.backward()
            self.tgp.clamp_grad()
            self.optimizer.step()
            self.tgp.clamp()
            self.append()
            if t % update_loss == 0:
                logp = self.tgp.logp()
                logp_median = logp.median().item()
                logp_std = (logp - logp_median).abs().median().item()
                desc = 'sgd | batch: {0:.2f}% | iters: [{1}, {2}) | logp: {3:.3f} ± {4:.3f} |'
                bar.set_description_str(desc.format(100 * self.nbatch / max(1, self.nobs), self.niters, end,
                                                    logp_median, logp_std))

    def execute_svgd(self, niters, update_loss=10, drop_niters=0):
        """
        Executes the SVGD algorithm.

        :param niters: an int, the number of iterations for the algorithm.
        :param update_loss: an int, the number of iterations to wait to update the loss value next to the progress bar.
        :param drop_niters: an int, the number of initial additional non-appended iterations.
        """
        bar = tqdm(range(niters), ncols=950)
        update_loss -= 1
        end = self.niters + niters
        logp = self.tgp.logp()
        no_grad = torch.no_grad()
        with no_grad:
            sigma = []
            for name, prior in self.priors_dict.items():
                for n, dist in prior.d.items():
                    sigma.append(dist.high - dist.low)
            sigma = torch.stack(sigma)
            epoch = int(niters * self.cycle)
        for t in range(drop_niters):
            logp = self.tgp.logp(index=self.index_batch)
            logp_nan = torch.isfinite(logp)
            loss = -logp[logp_nan].sum()
            self.optimizer.zero_grad()
            loss.backward()
            with no_grad:
                x = torch.stack([p if p.shape[0] > 1 else p.repeat(self.nparams, p.shape[1])
                                 for p in self.parameters_list], dim=1)
                dlogp = torch.stack([p.grad.data if p.shape[0] > 1 else p.grad.data.repeat(self.nparams, p.shape[1])
                                     for p in self.parameters_list], dim=1)
                dlogp = torch.clamp(dlogp, -self.dlogp_clamp, self.dlogp_clamp)

                d = self.svgd_direction(x, dlogp, sigma=sigma, annealing=0)
                if self.rand_pert:
                    for name, prior in self.priors_dict.items():
                        for n, dist in prior.d.items():
                            prior.p[n].data += (dist.high - dist.low) * torch.randn(prior.p[n].data.shape,
                                                                                    device=d.device) * self.rand_pert
                    #d = d + sigma * torch.randn(d.shape, device=d.device) * self.rand_pert
                for i, p in enumerate([p for p in self.parameters_list]):
                    p.grad.data = (d.data[:, i] if p.shape[0] > 1 else d.data[:, i].mean(dim=0, keepdim=True))
                self.tgp.clamp_grad()
                self.optimizer.step()
                self.tgp.clamp()

        for t in bar:
            logp = self.tgp.logp(index=self.index_batch)
            logp_nan = torch.isfinite(logp)
            loss = -logp[logp_nan].sum()
            self.optimizer.zero_grad()
            loss.backward()
            with no_grad:
                x = torch.stack([p if p.shape[0] > 1 else p.repeat(self.nparams, p.shape[1])
                               for p in self.parameters_list], dim=1)
                dlogp = torch.stack([p.grad.data if p.shape[0] > 1 else p.grad.data.repeat(self.nparams, p.shape[1])
                                   for p in self.parameters_list], dim=1)
                dlogp = torch.clamp(dlogp, -self.dlogp_clamp, self.dlogp_clamp)

                annealing_svgd = (((t % epoch) + 1) / epoch) ** self.pot if epoch > 0 else 1
                d = self.svgd_direction(x, dlogp, sigma=sigma, annealing=annealing_svgd)
                if self.rand_pert:
                    for name, prior in self.priors_dict.items():
                        for n, dist in prior.d.items():
                            prior.p[n].data += (dist.high - dist.low) * torch.randn(prior.p[n].data.shape,
                                                                                    device=d.device) * self.rand_pert
                    #d = d + sigma * torch.randn(d.shape, device=d.device) * self.rand_pert
                for i, p in enumerate([p for p in self.parameters_list]):
                    p.grad.data = (d.data[:, i] if p.shape[0] > 1 else d.data[:, i].mean(dim=0, keepdim=True))
                self.tgp.clamp_grad()
                self.optimizer.step()
                self.tgp.clamp()
            self.append()
            if t % update_loss == 0:
                self.optimizer.__init__(self.parameters_list, lr=self.optimizer.param_groups[0]['lr'])
                logp = self.tgp.logp()
                logp_median = logp.median().item()
                logp_std = (logp - logp_median).abs().median().item()
                desc = 'svgd | batch: {0:.2f}% | iters: [{1}, {2}) | logp: {3:.3f} ± {4:.3f} |'
                bar.set_description_str(desc.format(100 * self.nbatch / max(1, self.nobs), self.niters, end,
                                                    logp_median, logp_std))

    @staticmethod
    def svgd_direction(params: torch.Tensor, dlogp: torch.Tensor, sigma: torch.Tensor, annealing: torch.Tensor):
        """
        Calculate the kernel Stein direction with Gaussian kernel.

        :param params: a torch.tensor, the parameters of models.
        :param dlogp: a torch.tensor, the derivative of logp w.r. the parameters of models.
        :param sigma: a torch.tensor, the sigma of Gaussian kernel.
        :param annealing: a float, the annealing ponderator.

        :return: a torch.tensor, the kernel Stein direction-
        """
        n_samples = params.shape[0]
        d = (params[:, None, :] - params[None, :, :]) / sigma
        dists = d.pow(2).sum(axis=-1)
        h = 0.5 * dists.median().item() / log(n_samples + 1)
        h = h if h > 1e-3 else 1e-3
        k = torch.exp(- dists / (2 * h))
        k_der = d * k[:, :, None] / (sigma * h)

        ks = k.mm(dlogp)
        kd = k_der.sum(axis=0)
        return (annealing * ks + kd) / n_samples

    def append(self):
        pass

