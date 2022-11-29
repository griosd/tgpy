import torch
import torch.nn as nn
from .tensor import one, eps4, two, zero


class TgMapping(nn.Module):
    """
    This class implements the marginal transports that can be applied to stochastic processes. The forward is a function
    of t and x, i.e. h(t, x) = y, and have inverse wrt x, i.e. h^{-1}(t, y) = x, and log_gradient_inverse method.
    """
    def __init__(self, *args, **kwargs):
        """
        To define a marginal you do not need additional arguments.

        :param args: arguments to be passed to the class NgModule.
        :param kwargs: keyword arguments to be passed to the class NgModule.
        """
        # noinspection PyArgumentList
        super(TgMapping, self).__init__(*args, **kwargs)
        self.noise = False

    def __matmul__(self, other):
        """
        Implements the __matmul__ method, to compose two Marginal.

        :param other: another Marginal.

        :return: the same Composition object with updated number of operations.
        """
        return MarginalComposition(self, other)

    def forward(self, x, h):
        """
        A marginal transport needs a forward function that depends on time t and a distribution (empiric samples) x.

        :param x: a torch.Tensor, a time instant.
        :param h: a torch.Tensor, the distribution (empiric samples).
        """
        raise NotImplementedError

    def inverse(self, x, y):
        """
        A marginal transport needs the inverse function of the forward.

        :param x: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.
        """
        raise NotImplementedError

    def log_gradient_inverse(self, x, y):
        """
        A marginal transport needs to implement the logarithm of the gradient (with respect to y) of the inverse.

        :param t: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.
        """
        dy_p = self.inverse(x, y * (one + eps4) + eps4)
        dy_n = self.inverse(x, y * (one - eps4) - eps4)
        return torch.log((dy_p - dy_n) / (two * eps4 * (y + one)))

    def set_noise(self, noise):
        """
        Setting the noise parameter.

        :param noise: a boolean.
        """
        self.noise = noise

    @property
    def nargs(self):
        """Returns the number of arguments."""
        return 2


class MarginalComposition(TgMapping):
    """This class implements the marginal composition h(t, x) = h0(t, h1(t, x))."""
    def __init__(self, m0, m1, *args, **kwargs):
        """
        To compose you need two Marginal.

        :param m0: an Marginal.
        :param m1: another Marginal.
        :param args: arguments to be passed to the class Marginal.
        :param kwargs: keyword arguments to be passed to the class Marginal.
        """
        # noinspection PyArgumentList
        super(MarginalComposition, self).__init__(*args, **kwargs)
        self.m0 = m0
        self.m1 = m1

    def forward(self, x, h):
        """
        Computes the pushforward h(t, x) of the composition.

        :param x: a torch.Tensor, a time instant.
        :param h: a torch.Tensor, the distribution (empiric samples).

        :return: The pushforward of the composition
        """
        return self.m0(x, self.m1(x, h))

    def inverse(self, x, y):
        """
        Computes the inverse of h(t, x) of the composition.

        :param t: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: The inverse of the composition
        """
        return self.m1.inverse(x, self.m0.inverse(x, y))

    def log_gradient_inverse(self, x, y):
        """
        Computes the logarithm of the gradient (with respect to y) of inverse of h (the log-derivative of the inverse)
        via chain rule.

        :param x: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: a torch.Tensor, the log-derivative of the inverse.
        """
        return self.m1.log_gradient_inverse(x, self.m0.inverse(x, y)) + self.m0.log_gradient_inverse(x, y)

    def set_noise(self, noise):
        """
        Setting the noise parameter.

        :param noise: a boolean.
        """
        self.noise = noise
        self.m0.set_noise(noise)
        self.m1.set_noise(noise)


class LogShift(TgMapping):
    """This class implements the marginal transport h(t, x) = shift(t) + e^x."""
    def __init__(self, shift, *args, **kwargs):
        """
        We need one function.

        :param shift: a function of t, to shift e^x.
        :param args: arguments to be passed to the Marginal class.
        :param kwargs: keyword arguments to be passed to the Marginal class.
        """
        # noinspection PyArgumentList
        super(LogShift, self).__init__(*args, **kwargs)
        self.shift = shift

    def forward(self, x, h):
        """
        Computes the pushforward h(t, x).

        :param x: a torch.Tensor, a time instant.
        :param h: a torch.Tensor, the distribution (empiric samples).

        :return: The pushforward shift(t) + e^x.
        """
        return self.shift()[:, None, :] + torch.exp(h)

    def inverse(self, x, y):
        """
        Computes the inverse of h(t, x) with respect to x.

        :param x: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: The inverse of h: log(y - shift(t)).
        """
        return torch.log(y[None, :, :] - self.shift()[:, None, :])

    def log_gradient_inverse(self, x, y):
        """
        Computes the logarithm of the gradient (with respect to y) of inverse of h (the log-derivative of the inverse).

        :param x: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: a torch.Tensor, the log-derivative of the inverse.
        """
        return -(y[None, :, :] - self.shift()[:, None, :]).log()


class LogitScale(TgMapping):
    """This class implements the marginal transport h(t, x) = l + u/(1 + e^(-scale(t) * (x-shift(t))))."""
    def __init__(self, scale, shift, lower=0, upper=1, *args, **kwargs):
        """
        We need two functions and two parameters.

        :param scale: a function of t, to scale (x - shift(t))
        :param shift: a function of t, to shift x.
        :param lower: a float, the lower bound.
        :param upper: a float, the upper bound.
        :param args: arguments to be passed to the Marginal class.
        :param kwargs: keyword arguments to be passed to the Marginal class.
        """
        # noinspection PyArgumentList
        super(LogitScale, self).__init__(*args, **kwargs)
        self.scale = scale
        self.shift = shift
        self.lower = lower
        self.upper = upper

    def forward(self, x, h):
        """
        Computes the pushforward h(t, x).

        :param x: a torch.Tensor, a time instant.
        :param h: a torch.Tensor, the distribution (empiric samples).

        :return: The pushforward  l + u/(1 + e^(-scale(t) * (x-shift(t)))).
        """
        return self.lower + self.upper / (1 + torch.exp(-self.scale() * (h - self.shift())))

    def inverse(self, x, y):
        """
        Computes the inverse of h(t, x) with respect to x.

        :param x: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: The inverse of h: shift(t) - log(u/(y-l) - 1)/scale(t).
        """
        return self.shift() - torch.log(self.upper / (y - self.lower) - 1) / self.scale()
        # self.shift(t)+(torch.log((y-self.lower)/self.upper)-torch.log(1-(y+self.lower)/self.upper))/self.scale(t)

    def log_gradient_inverse(self, x, y):
        """
        Computes the logarithm of the gradient (with respect to y) of inverse of h (the log-derivative of the inverse).

        :param x: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: a torch.Tensor, the log-derivative of the inverse.
        """
        return torch.log(self.upper / (self.upper - y + self.lower)) - torch.log(self.scale())
        # -torch.log((y-self.lower)/self.upper) - torch.log(1-(y+self.lower)/self.upper)-torch.log(self.scale(t))


class BoxCoxShift(TgMapping):
    """
    This class implements the marginal transport ``h(t, x) = shift(t) + sign(1 + x*power) * |1 + x*power|^(1/power)``.
    """
    def __init__(self, power, shift, *args, **kwargs):
        """
        We need two functions.

        :param power: a function of t, a positive float.
        :param shift: a function of t, to shift the Box-Cox transformation.
        :param args: arguments to be passed to the Marginal class.
        :param kwargs: keyword arguments to be passed to the Marginal class.
        """
        # noinspection PyArgumentList
        super(BoxCoxShift, self).__init__(*args, **kwargs)
        self.shift = shift
        self.power = power

    def forward(self, x, h):
        """
        Computes the pushforward h(t, x).

        :param x: a torch.Tensor, a time instant.
        :param h: a torch.Tensor, the distribution (empiric samples).

        :return: The pushforward ``shift(t) + sign(1 + x*power) * |1 + x*power|^(1/power)``.
        """
        power = self.power()[:, None, :]
        scaled = (power * h) + one
        transformed = torch.sign(scaled) * (torch.abs(scaled) ** (one / power))
        return self.shift()[:, None, :] + transformed

    def inverse(self, x, y):
        """
        Computes the inverse of h(t, x) with respect to x.

        :param x: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: The inverse of h: ``(sign(y-shift(t)) * |y-shift(t)|^power - 1) / power``.
        """
        power = self.power()[:, None, :]
        shifted = y[None, :, :] - self.shift()[:, None, :]
        r = ((torch.sign(shifted) * torch.abs(shifted) ** power) - one) / power
        return r

    def log_gradient_inverse(self, x, y):
        """
        Computes the logarithm of the gradient (with respect to y) of inverse of h (the log-derivative of the inverse).

        :param x: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: a torch.Tensor, the log-derivative of the inverse.
        """
        shifted = y[None, :, :] - self.shift()[:, None, :]
        power = self.power()[:, None, :]
        return (power - one) * torch.max(torch.abs(shifted), y * zero + eps4).log()


class Relu(TgMapping):
    """This class implements the marginal transport h(t, x) = scale(t) * max(0, x)."""
    def __init__(self, scale=one, shift=zero, *args, **kwargs):
        """
        We need one function.

        :param scale: a function of t, to shift the ReLU.
        :param shift: a constant to shift the ReLU.
        :param args: arguments to be passed to the Marginal class.
        :param kwargs: keyword arguments to be passed to the Marginal class.
        """
        # noinspection PyArgumentList
        super(Relu, self).__init__(*args, **kwargs)
        self.scale = scale
        self.shift = shift

    def forward(self, t, x):
        """
        Computes the pushforward h(t, x).

        :param t: a torch.Tensor, a time instant.
        :param x: a torch.Tensor, the distribution (empiric samples).

        :return: The pushforward scale(t) * max(0, x).
        """
        if self.noise:
            return (x.add(self.shift)) * self.scale(t)
        else:
            return torch.nn.functional.relu(x.add(self.shift)) * self.scale

    def inverse(self, t, y):
        """
        Computes the inverse of h(t, x) with respect to x.

        :param t: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: The inverse of h: y / scale(t).
        """
        return y.div(self.scale).sub(self.shift)

    def log_gradient_inverse(self, t, y):
        """
        Computes the logarithm of the gradient (with respect to y) of inverse of h (the log-derivative of the inverse).

        :param t: a torch.Tensor, a time instant.
        :param y: a torch.Tensor.

        :return: a torch.Tensor, the log-derivative of the inverse.
        """
        return -torch.ones_like(y[None, :]).mul(self.scale.log())
