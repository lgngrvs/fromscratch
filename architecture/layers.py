import tools as tls # See tools.py
from tools import Module, Parameter, Tensor, ModuleList
from collections.abc import Callable
from einops import einsum

"""
layers.py

Basic layer components composed to create larger models.

TODO:
- Learn more about initialization; right now weight layers just use kaiming initialization and biases just set to zero (possibly dumb?) by default because I heard they were good once, and I'm not really optimizing yet. At some point I would like to roll my own initializations.
- Maybe create a helper function for weight initialization?
- Add a linter to repo so that function arguments get prettified.
- Create a ValidationHarness in a separate file that runs things against their torch originals
"""


"""
======== NONLINEARITIES ========
"""

def ReLU(x: Tensor):
    """
    Literally ReLU.
    """
    mask = x < 0
    x[mask] = 0
    return x

def softmax(x: Tensor, dim: int, masked: bool=False, mask_dim: int=None) -> Tensor:
    """
    For vector z, softmax(z_i) = e^(z_i) / (sum over e^(z_j))
    So Softmax on a tensor picks a single dimension, and then 
    softmaxes all the vectors on that dimension.
    """
    dimensions = list(x.size())
    # print(f"x: {x}") 
    
    if masked:
        assert mask_dim is not None, "masking is set to true so mask_dim must be provided"
        mask_dims = [x.size(dim=mask_dim), x.size(dim=dim)] # get 2d mask shape
        inf_mask = tls.triu(tls.ones(mask_dims), diagonal=1) > 0 # triu with diag=1 gives you 1s, 1 above the regular diagonal; get binary mask using > 0 (entries >0 return true)
        # print(f"inf mask: {inf_mask}")
        x.masked_fill_(inf_mask, -tls.inf)
        # print(f"masked x: {x}")

    x_exp = tls.exp(x) # exponentiates all the elements
    exp_sum = tls.sum(x_exp, dim=dim, keepdim=True) # sums out the given dimension
    out = x_exp * tls.reciprocal(exp_sum) # flip to divide by the sum. will broadcast
    return out


"""
======== MODULES ========
"""

class Linear(Module):
    """
    Technically affine. Basic single-layer matmul + bias.
    """
    def __init__(self, dim_1: int, dim_2: int):
        super(Linear, self).__init__()
        self.w = Parameter(Tensor(dim_1, dim_2))
        self.b = Parameter(Tensor(dim_2,))
        tls.init.kaiming_normal_(self.w)
        tls.init.zeros_(self.b)

    def forward(self, x: Tensor) -> Tensor:
        out = einsum(x, self.w, "... a, a b -> ... b") + self.b
        return out

class MLP(Module):
    """
    MLP. Composes arbitrary numbers of linear layers in arbitrary dimensions 
    """
    def __init__(self, num_layers: int, dimensions:list[int], activ_func:Callable[[Tensor], Tensor] = ReLU):
        super(MLP, self).__init__()
        assert len(dimensions) == num_layers + 1, "Dimensions list must have length == num_layers + 1" 
        self.layers = ModuleList([Linear(dimensions[l], dimensions[l+1]) for l in range(num_layers)])
        self.n_layers = len(self.layers)
        self.activ_func = activ_func

    def forward(self, x: Tensor) -> Tensor:
        for l in range(self.n_layers):
            print(x)
            x = self.layers[l](x)
            print(x)
            if l < self.n_layers - 1:
                x = self.activ_func(x)
        return x

class MultiHeadAttention(Module):
    """
    Standard MHA; causal masking is on by default
    and can be turned off.
    """
    def __init__(self, latent_dim: int, seq_len: int, qk_dim: int, n_heads: int, v_dim: int = None, causal_mask: bool = False):
        super(MultiHeadAttention, self).__init__()
        assert n_heads > 0, "n_heads must be positive" 
        if v_dim is None:
            v_dim = int(round(latent_dim/seq_len))
            
        self.latent_dim, self.seq_len, self.qk_dim, self.n_heads, self.v_dim = latent_dim, seq_len, qk_dim, n_heads, v_dim # Surely there is a better way to unpack these args

        self.W_q = Parameter(Tensor(latent_dim, n_heads, qk_dim))
        self.W_k = Parameter(Tensor(latent_dim, n_heads, qk_dim))
        self.W_v = Parameter(Tensor(latent_dim, n_heads, v_dim))
        o_dim = v_dim * n_heads
        self.W_o = Parameter(Tensor(o_dim, latent_dim))
        for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
            tls.init.kaiming_normal_(w)


    def forward(self, x: Tensor):
        print(f"x size: {x.size()}, self.W_q size: {self.W_q.size()}")
        # x size: torch.Size([2, 3, 4]), self.W_q size: torch.Size([3, 4, 2])
        Q = einsum(x, self.W_q, "... seq lat, lat heads dqk -> ... seq heads dqk")
        K = einsum(x, self.W_k, "... seq lat, lat heads dqk -> ... seq heads dqk")
        V = einsum(x, self.W_v, "... seq lat, lat heads dv -> ... seq heads dv") 

        QK = einsum(Q, K, "... seq1 heads dqk, ... seq2 heads dqk -> ... heads seq1 seq2") # I don't think I actually get how this one works. Einsum is letting me get away with shit.
        QK_dk = QK * (tls.ones_like(QK) * tls.sqrt(tls.tensor(self.qk_dim)))
        attention_scores = softmax(QK_dk, dim=-1, masked=True, mask_dim=-2)
        sQKV = einsum(attention_scores, V, "... heads seq seq, ... seq heads dv -> ... seq heads dv")
        out = einsum(sQKV, self.W_o, "... seq o_dim, o_dim latent_dim-> ... seq latent_dim") 
        return out

"""
======== BASIC TESTS ========   

Just running the components on
simple inputs to make sure there
are no immediate errors.
"""

def run_basic_tests():
    x = tls.rand(4,2,3)
    y_0 = softmax(x, dim=-1, masked=True, mask_dim=-2)
    # print("Softmax test: ", y_0)

    x = tls.ones(2,2)
    layer = Linear(2, 2)
    y_1 = layer(x)
    # print("Linear test: ", y_1)

    mlp = MLP(2, [2,3,2])
    y_2 = mlp(x)
    # print("MLP test: ", y_2)

    latent_dim = 3
    seq_len = 4
    mha = MultiHeadAttention(latent_dim, seq_len, 2, 4)
    x = tls.rand(2, seq_len, latent_dim)
    y_3 = mha(x)
    print("MHA test: ", y_3)

    print("Nice job, no errors!")


if __name__ == "__main__":
    run_basic_tests()
