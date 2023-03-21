import torch
import torch.nn as nn


class TgRadial(nn.Module):
    """
    phi(||x||) x, phi(r) = alpha(r) / r.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(TgRadial, self).__init__(*args, **kwargs)

    def forward(self, t, x):
        raise NotImplementedError

    def inverse(self, t, y):
        raise NotImplementedError

    def log_gradient_inverse(self, t, y):
        # log det inverse S_t (y) = (n - 1) log [alpha^(-1)||y||]  - log [alpha' (alpha^{-1} (||y||)) ]
        pass

    def __matmul__(self, other):
        """
        Implements the __matmul__ method, to compose two Radial Maps.

        :param other: another Marginal.

        :return: the same Composition object with updated number of operations.
        """
        return MarginalComposition(self, other)

class CustomRadial(TgRadial):
    # siempre hacer el caso l2
    def __init__(self, radial_map, inv_radial_map, norm_type='l2', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.radial_map = radial_map # alpha
        self.inv_radial_map = inv_radial_map
        # obtiain phi = alpha(r) / r

        if norm_type.lower() !=  'l2':
            raise NotImplementedError
        self.norm_type = norm_type

    def forward(self, t, x):
        # apply mapping phi(||x||) x taking into account the norm
        pass

    def inverse(self, t, y):
        pass


class DummyRadial(TgRadial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # obtiain phi = alpha(r) / r

    def forward(self, t, x):
        # apply mapping phi(||x||) x taking into account the norm
        return x

    def inverse(self, t, y):
        return y

    def log_gradient_inverse(self, t, y):
        # log det inverse S_t (y) = (n - 1) log [alpha^(-1)||y||]  - log [alpha' (alpha^{-1} (||y||)) ]
        pass


class StudentTRadial(TgRadial):
    def __init__(self, *args, **kwargs) -> None:
        # TODO: add student-t parameters
        super(TgRadial, self).__init__()
        self.radial_map = None # TODO(ale) given the parameters create the atribute with the map

    def forward(self, x):
        # implement particular case forward.
        pass
