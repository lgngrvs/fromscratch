# imports file --- explicitly contains all the stuff I'm allowed to use and don't need to reimplement. Things can be removed from this list as I implement them myself.

from torch import Tensor, einsum, ones
# Allowed to have Module and Parameter
from torch.nn import Module, Parameter, ModuleList
# init stuff is contained in tools/init
from . import init # this imports kaiming and zeros
