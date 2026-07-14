# imports file --- explicitly contains all the stuff I'm allowed to use and don't need to reimplement. Things can be removed from this list as I implement them myself.

# using einsum from einops instead of torch einsum
from torch import Tensor, ones, ones_like, sqrt, tensor, exp, sum, reciprocal, triu, tril, rand, inf, flatten, mean, var, zeros, empty, randint, cat, log, zeros_like, argmax, max, full, numel, no_grad, stack, arange, sin, cos

# Allowed to have Module and Parameter
from torch.nn import Module, Parameter, ModuleList  # noqa: F401
from torch.nn.functional import one_hot
# init stuff is contained in tools/init
from . import init # this imports kaiming and zeros  # noqa: F401
