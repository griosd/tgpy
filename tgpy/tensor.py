import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.autograd import Function
from scipy import special

if 'CUDA' not in os.environ:
    os.environ['CUDA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if hasattr(torch.backends, 'cudnn'):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.nn.Module.dump_patches = False
_device = 'cuda' if (torch.cuda.is_available() and os.environ['CUDA'] == '1') else 'cpu'
zero = torch.tensor(0.0, device=_device)
one = torch.tensor(1.0, device=_device)
two = torch.tensor(2.0, device=_device)
eps4 = torch.tensor(1e-4, device=_device)

dim_index = 0
dim_series = 1
dim_gram = 2


class DataTensor:
    def __init__(self, df, inputs, outputs, calculate_shift=True, calculate_scale=True, device='cuda'):
        self.df = df
        self.inputs = inputs
        self.outputs = outputs
        self.device = device
        self.shift_inputs = 0
        self.shift_outputs = 0
        self.scale_inputs = 1
        self.scale_outputs = 1
        if calculate_shift:
            self.calculate_shift()
        if calculate_scale:
            self.calculate_scale()

    @property
    def index(self):
        return self.df.index

    @property
    def ndata(self):
        return len(self.df.index)

    def _repr_html_(self):
        return pd.concat([self.original_inputs(), self.original_outputs()], axis=1)._repr_html_()

    def calculate_shift(self, inputs=True, outputs=True, stat='min'):
        if inputs:
            self.shift_inputs = self.original_inputs().apply(stat)
        if outputs:
            self.shift_outputs = self.original_outputs().apply(stat)

    def calculate_scale(self, inputs=True, outputs=True, quantile=0.9):
        if inputs:
            self.scale_inputs = self.df_inputs().quantile(quantile)
        if outputs:
            self.scale_outputs = self.df_outputs().quantile(quantile)

    def original(self, index=Ellipsis):
        return self.df[self.inputs + self.outputs].loc[index]

    def original_inputs(self, index=Ellipsis):
        return self.df[self.inputs].loc[index]

    def original_outputs(self, index=Ellipsis):
        return self.df[self.outputs].loc[index]

    def df_inputs(self, index=Ellipsis):
        return (self.df[self.inputs].loc[index]-self.shift_inputs)/self.scale_inputs

    def df_outputs(self, index=Ellipsis):
        return (self.df[self.outputs].loc[index]-self.shift_outputs)/self.scale_outputs

    def tensor_inputs(self, index=Ellipsis):
        return self.df_to_tensor(self.df_inputs(index))

    def tensor_outputs(self, index=Ellipsis):
        return self.df_to_tensor(self.df_outputs(index))

    def df_to_tensor(self, df):
        return torch.Tensor(df.values).to(self.device)

    def tensor_to_df_outputs(self, t):
        return pd.DataFrame(to_numpy(t))

    def tensor_to_df_inputs(self, t):
        return pd.DataFrame(to_numpy(t))

    def tensor_to_original_outputs(self, t):
        return pd.DataFrame(to_numpy(t)) * self.scale_outputs.item() + self.shift_outputs.item()

    def tensor_to_original_inputs(self, t):
        return pd.DataFrame(to_numpy(t)) * self.scale_inputs.item() + self.shift_inputs.item()

    def original_to_tensor_inputs(self, df):
        return self.df_to_tensor((df-self.shift_inputs)/self.scale_inputs)

    def get_grid(self, delta=0, n=100):
        stats = self.original_inputs().describe()
        min_x = stats.loc['min', self.inputs[0]]
        max_x = stats.loc['max', self.inputs[0]]
        delta_x = (max_x - min_x) * delta
        min_y = stats.loc['min', self.inputs[1]]
        max_y = stats.loc['max', self.inputs[1]]
        delta_y = (max_y - min_y) * delta
        space_e = np.linspace(min_x - delta_x, max_x + delta_x, n)
        space_n = np.linspace(min_y - delta_y, max_y + delta_y, n)
        space_e, space_n = np.meshgrid(space_e, space_n)
        space = pd.DataFrame(np.vstack([space_e.reshape(-1), space_n.reshape(-1)]).T, columns=self.inputs[:2])
        return space


def to_numpy(tensor):
    """
    Returns an np.array() of a torch.Tensor.

    :param tensor: a torch.Tensor.

    :return: a numpy array with the same tensor data.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, pd.Series) or isinstance(tensor, pd.DataFrame):
        return tensor.values
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().data.cpu().numpy()
    else:
        return np.float32(tensor)


def to_tensor(array, device=None):
    """
    Converts array or float to torch.Tensor

    :param array: a numpy array or float.
    :param device: a string, the device for the tensor.

    :return: a torch tensor.
    """
    if isinstance(array, np.array) or isinstance(array, list):
        return torch.Tensor(array, device=device)
    elif isinstance(array, float):
        return torch.ones(1, device=device) * array
    elif isinstance(array, torch.Tensor):
        return array
    else:
        print("Argument array could not be converted to tensor.")
        return None


def cholesky(matrix, upper=False, out=None, jitter=None):
    """
    Returns the Cholesky decomposition of A.

    :param matrix: a torch.tensor, the matrix to which the decomposition will be computed.
    :param upper: a boolean, if True the decomposition will be A=U^TU, if False decomposition will be A=LL^T
    :param out: a torch.tensor, the output matrix
    :param jitter: a float to add to the diagonal in case A is only p.s.d. If omitted, it is chosen as 1e-6 (float) or
        1e-8 (double)

    :return: the Cholesky decomposition of A.
    """
    return robust_cholesky(matrix.view(-1, matrix.shape[-2], matrix.shape[-1]).contiguous(),
                           upper, out, jitter).view(matrix.shape)


def robust_cholesky(matrix, upper=False, out=None, jitter=None):
    """
    Computes the Cholesky decomposition of A. If A is only positive semidefinite (p.s.d), it adds a small jitter to the
    diagonal.

    :param matrix: a torch.Tensor, the matrix to which the decomposition will be computed.
    :param upper: a boolean, whether to return the decomposition as A=U^TU or as A=LL^T.
    :param out: a torch.Tensor, the output matrix.
    :param jitter: a float to add to the diagonal in case A is only p.s.d. If omitted, it is chosen as 1e-6 (float) or
        1e-8 (double).

    :return: the Cholesky decomposition of A.
    """
    try:
        # robust_zero(A)
        cho = torch.linalg.cholesky(matrix, upper=upper, out=out)
        return cho
    except RuntimeError:
        _robust_zero(matrix)
        matrix_max = matrix.diagonal(dim1=1, dim2=2).mean(dim=1)[:, None, None]
        matrix = matrix/matrix_max
        jitter = jitter if jitter else 1e-6 if matrix.dtype == torch.float32 else 1e-8
        jitter_ori = jitter
        for i in range(5):
            jitter = jitter_ori * (10 ** i)
            matrix.diagonal(dim1=-2, dim2=-1).add_(jitter)
            try:
                cho = torch.linalg.cholesky(matrix, upper=upper, out=out)
                return cho * (matrix_max ** 0.5)
            except RuntimeError:
                continue
        return matrix * 1e-6


@torch.jit.script
def _robust_zero(matrix):
    """
    Fills (inplace) the NaN values of matrix a with a small number.

    :param matrix: a torch.Tensor.
    """
    for i in range(matrix.shape[0]):
        if matrix[i, 0, 0] != matrix[i, 0, 0]:
            matrix[i].fill_diagonal_(1e-10)


def gpu_clean():
    """Cleans all unoccupied cached memory currently held by the caching allocator in the GPU."""
    gc.collect()
    del gc.garbage[:]
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()


class TorchGammaincinv(Function):
    @staticmethod
    def forward(ctx, a, y):
        ctx.save_for_backward(a, y)
        return torch.as_tensor(special.gammaincinv(to_numpy(a), to_numpy(y)), device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps=1e-4):
        a, y = ctx.saved_tensors
        grad_a = grad_y = None
        if ctx.needs_input_grad[0]:  # replace a by grad_output
            # grad_a = 1 / TorchGammainc.backward(ctx, TorchGammaincinv.apply(a, y))[0]
            grad_a = grad_output * (TorchGammaincinv.apply(a * (1+eps), y) - TorchGammaincinv.apply(a * (1-eps),
                                                                                                    y)) / (2*a*eps)
        if ctx.needs_input_grad[1]:  # replace y by grad_output
            # grad_y = 1 / TorchGammainc.backward(ctx, TorchGammaincinv.apply(a, y))[1]
            grad_y = grad_output*(TorchGammaincinv.apply(a, y*(1+eps)) - TorchGammaincinv.apply(a,
                                                                                                y*(1-eps))) / (2*y*eps)
        return grad_a, grad_y


gammaincinv = TorchGammaincinv.apply


class TorchBetainc(Function):
    @staticmethod
    def forward(ctx, a, b, x):
        ctx.save_for_backward(a, b, x)
        return torch.as_tensor(special.betainc(to_numpy(a), to_numpy(b), to_numpy(x)), device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps=1e-4):
        a, b, x = ctx.saved_tensors
        grad_a = grad_b = grad_x = None
        if ctx.needs_input_grad[0]:  # replace a by grad_output
            grad_a = grad_output*(TorchBetainc.apply(a*(1+eps), b, x)-TorchBetainc.apply(a*(1-eps), b, x))/(2*a*eps)
        if ctx.needs_input_grad[1]:  # replace b by grad_output
            grad_b = grad_output*(TorchBetainc.apply(a, b*(1+eps), x) - TorchBetainc.apply(a, b*(1-eps), x)) / (2*b*eps)
        if ctx.needs_input_grad[2]:  # replace x by grad_output
            grad_x = grad_output*(torch.lgamma(a+b)-torch.lgamma(a)-torch.lgamma(b)).exp() * \
                     torch.pow(x, a-1)*torch.pow(1-x, b-1)
        return grad_a, grad_b, grad_x


betainc = TorchBetainc.apply


class TorchBetaincinv(Function):
    @staticmethod
    def forward(ctx, a, b, y):
        ctx.save_for_backward(a, b, y)
        return torch.as_tensor(special.betaincinv(to_numpy(a), to_numpy(b), to_numpy(y)), device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output, eps=1e-4):
        a, b, y = ctx.saved_tensors
        grad_a = grad_b = grad_y = None
        if ctx.needs_input_grad[0]:  # replace a by grad_output
            grad_a = grad_output*(TorchBetaincinv.apply(a*(1+eps), b, y) - TorchBetaincinv.apply(a*(1-eps),
                                                                                                 b, y))/(2*a*eps)
        if ctx.needs_input_grad[1]:  # replace b by grad_output
            grad_b = grad_output*(TorchBetaincinv.apply(a, b*(1+eps), y) - TorchBetaincinv.apply(a, b*(1-eps),
                                                                                                 y))/(2*b*eps)
        if ctx.needs_input_grad[2]:  # replace y by grad_output
            grad_y = grad_output*(TorchBetaincinv.apply(a, b, y*(1+eps)) - TorchBetaincinv.apply(a, b,
                                                                                                 y*(1-eps)))/(2*y*eps)
        return grad_a, grad_b, grad_y


betaincinv = TorchBetaincinv.apply
