import tools as tls
from tools import Module, Parameter, Tensor, ModuleList
from collections.abc import Callable
from einops import einsum

def ReLU(x: Tensor):
    mask = x < 0
    x[mask] = 0
    return x

def Softmax(x: Tensor, dim=int):
    pass

"""
TODO:
- implement flexible softmax (right now just returns x)
"""

class Linear(Module):
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


"""
MultiHeadAttention:
    Weights: W_q, W_k, W_v, W_o
    x is shape b,s,h
    W_q, W_k = (h, s)
    Softmax(QK^T/d_k)V 
    take in batch size, seq_length, hidden_dim
    then attend over seq_length with mask at hidden_dim, hidden_dim
    you get attention scores, then hidden_dim x 

    ok i got it after fumbling around a bit
    
"""

class MultiHeadAttention(Module):
    def __init__(self, latent_dim: int, seq_len: int, qk_dim: int, n_heads: int, v_dim: int = None):
        assert n_heads > 0, "n_heads must be positive" 
        if v_dim is None:
            v_dim = int(round(latent_dim/seq_len))
            
        self.latent_dim, self.seq_len, self.qk_dim, self,n_heads, self.v_dim = latent_dim, seq_len, qk_dim, n_heads, v_dim # Surely there is a better way to unpack these args

        W_q = Tensor(latent_dim, n_heads, qk_dim)
        W_k = Tensor(latent_dim, n_heads, qk_dim)
        W_v = Tensor(seq_len, n_heads, v_dim)
        o_dim = v_dim * n_heads
        W_o = Tensor(o_dim, latent_dim)

    def forward(self, x: Tensor):
        Q = einsum(x, self.W_q, "... seq lat, lat heads dqk -> ... seq heads dqk")
        K = einsum(x, self.W_k, "... seq lat, lat heads dqk -> ... seq heads dqk")
        V = einsum(x, self.W_v, "... seq lat, seq heads dv -> ... seq heads dv") # do i need to be careful here? maybe these indices need to be different to disambiguate? I think this is incorrect dimensions
        QK = einsum(Q, K, "... seq heads dqk, ... seq heads dqk -> ... heads seq seq")
        QK_dk = QK * (tls.ones_like(QK) * torch.sqrt(self.qk_dim)) # normalize
        s = Softmax(QK_dk, dim=-1) # make sure you end up using the right seq len now 
        sQKV = einsum(s, V, "... heads seq seq, seq heads dv -> ... seq heads dv") # incorrect dimensions
        
        # sQKV.rearrange() or whatever
        
        out = einsum(sQKV, self.W_o, "... seq o_dim, o_dim -> latent_dim") 
        return out

        # def some errors here! that's ok

x = tls.ones(2,2)
layer = Linear(2, 2)
y_1 = layer(x)
print("Linear test: ", y_1)

mlp = MLP(2, [2,3,2])
y_2 = mlp(x)
print("MLP test: ", y_2)


print("Nice job, no errors!")
