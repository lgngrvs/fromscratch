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
======== TOKENIZERS ======
"""

class Tokenizer():
    """
    Empty base class object.
    Tokenizers should be classes, not just objects, because they need to be trainable
    """
    pass
    

# is this a module?
class ToyTokenizer(Tokenizer):
    """
    Toy tokenizer for testing.
    - Vocab size 27, latent_size 27
    - accepts lowercase letters and spaces only
    - turns them into token ids 0,..., 26
    """
    def __init__(self):
        super(ToyTokenizer, self).__init__()
        self.vocab_size = 27
        
    def tokenize(self, s: str):
        allowed_letters = list("abcdefghijklmnopqrstuvwxyz ")
        split = list(s)
        assert set(split).issubset(set(allowed_letters)), "Toy Tokenizer only accepts lowercase letters or spaces."

        # turn string into list of ids
        id_list=[ allowed_letters.index(letter) for letter in split ]
        return id_list


class EmbeddingLayer(Module):
    def __init__(self, vocab_size: int, latent_dim: int):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size, self.latent_dim = vocab_size, latent_dim
        self.E = Parameter(Tensor(self.vocab_size, self.latent_dim))
        tls.init.kaiming_normal_(self.E)

    def forward(self, x: list) -> Tensor:
       return self.E[x] # just read the rows rather than doing one-hot multiplication (wasting matmuls)


class UnembeddingLayer(Module):
    def __init__(self, vocab_size: int, latent_dim: int, tied_weight: Parameter=None):
        super(UnembeddingLayer, self).__init__()
        if tied_weight is not None:
           self.E = tied_weight 
        else: 
            self.E = Parameter(Tensor(self.vocab_size, self.latent_dim))
            tls.init.kaiming_normal_(self.E)

    def forward(self, x: Tensor) -> Tensor:
       return einsum(x, self.E.t(), "... latent_dim, latent_dim vocab_size -> ... vocab_size")



class ToyEmbedding(EmbeddingLayer): # aaaaagh that's not how embeddings work lol bruh it's just another layer
    """
    Takes in token IDs from ToyTokenizer
    and trivially turns them into one-hots
    with 1 in the token id place.

    Should be a way to indicate it only takes stuff
    from ToyTokenizer/ they fit together? IDK
    Originally these were one component, but
    I split them in two so we could have explicit
    tokenizer/embed separations.
    """
    def __init__(self):
        super(ToyEmbedding, self).__init__()
        self.vocab_size = 27
        self.latent_dim = 27

    def embed(self, ids: list):
        assert set(ids).issubset(set([i for i in range(27)])), "Toy Embedding only takes token ids from ToyTokenizer, with indices 0-26 inclusive."
        embedded_tensor = tls.zeros(len(ids), 27) # All-zeros tensor
        for seq_pos, idx, in enumerate(ids):
            embedded_tensor[seq_pos, idx] = 1
        return embedded_tensor

        
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

class LayerNorm(Module):
    """
    Takes x of shape (batch, seq, latent)
    Just normalizes the latent at each position,
    and applies a learned scaling factor.
    """
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(tls.tensor(1.0))
        self.beta = Parameter(tls.tensor(0.0))
        
    def forward(self, x):
        eps = 1e-5
        mu = tls.mean(x, dim=-1, keepdims=True)
        sigma = tls.sqrt(tls.var(x, dim=-1, keepdims=True) + eps)
        normalized =  self.gamma * ((x - mu) / sigma) + self.beta
        return normalized
        
       

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
            # print(x)
            x = self.layers[l](x)
            # print(x)
            if l < self.n_layers - 1:
                x = self.activ_func(x)
        return x

class MultiHeadAttention(Module):
    """
    Standard MHA; causal masking is on by default
    and can be turned off.
    """
    def __init__(self, latent_dim: int, seq_len: int, qk_dim: int, n_heads: int, v_dim: int = None, causal_mask: bool = True):
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
        # print(f"x size: {x.size()}, self.W_q size: {self.W_q.size()}")
        # x size: torch.Size([2, 3, 4]), self.W_q size: torch.Size([3, 4, 2])
        Q = einsum(x, self.W_q, "... seq lat, lat heads dqk -> ... seq heads dqk")
        K = einsum(x, self.W_k, "... seq lat, lat heads dqk -> ... seq heads dqk")
        V = einsum(x, self.W_v, "... seq lat, lat heads dv -> ... seq heads dv") 

        QK = einsum(Q, K, "... seq1 heads dqk, ... seq2 heads dqk -> ... heads seq1 seq2") # I don't think I actually get how this one works. Einsum is letting me get away with shit.
        QK_dk = QK * (tls.ones_like(QK) * tls.sqrt(tls.tensor(self.qk_dim)))
        attention_scores = softmax(QK_dk, dim=-1, masked=True, mask_dim=-2)
        sQKV = einsum(attention_scores, V, "... heads seq seq, ... seq heads dv -> ... seq heads dv").flatten(-2,-1) # flatten to concatenate heads back together
        out = einsum(sQKV, self.W_o, "... seq o_dim, o_dim latent_dim-> ... seq latent_dim") 
        return out

class TransformerBlock(Module):
    """
    Simple transformer with multihead self-attention and MLP blocks.
    Still needs an embed, vocab, etc. so right now it's just blocks.
    """
    def __init__(self, 
        latent_dim: int, 
        seq_len: int, 
        qk_dim: int, 
        n_heads: int, 
        num_mlp_layers: int, 
        mlp_dimensions:list[int], 
        activ_func:Callable[[Tensor], Tensor] = ReLU, 
        v_dim: int = None, 
        causal_mask: bool = False
    ):
        super(TransformerBlock, self).__init__()
        assert mlp_dimensions[-1] == latent_dim
        assert mlp_dimensions[0] == latent_dim


        self.latent_dim, self.seq_len, self.qk_dim, self.n_heads, self.num_mlp_layers, self.mlp_dimensions, self.activ_func, self.v_dim, self.causal_mask= latent_dim, seq_len, qk_dim, n_heads, num_mlp_layers, mlp_dimensions, activ_func, v_dim, causal_mask # Surely there is a better way to unpack these args
        self.mhsa = MultiHeadAttention(latent_dim, seq_len, qk_dim, n_heads, v_dim=v_dim, causal_mask=causal_mask)
        self.ln_1 = LayerNorm()
        self.mlp = MLP(num_mlp_layers, mlp_dimensions, activ_func=activ_func)
        self.ln_2 = LayerNorm()

    def forward(self, x):
        x += self.mhsa(x)
        x = self.ln_1(x)
        x += self.mlp(x)
        x = self.ln_2(x)
        return x

class StandardTransformer(Module):
    """
    A proper decoder-only transformer.
    """
    def __init__(self, 
            num_blocks: int,
            tokenizer: Tokenizer, # should have vocab_size in it 
            latent_dim: int, 
            seq_len: int, 
            qk_dim: int, 
            n_heads: int, 
            num_mlp_layers: int, 
            mlp_dimensions:list[int], 
            activ_func:Callable[[Tensor], Tensor] = ReLU, 
            v_dim: int = None, 
            causal_mask: bool = False
    ):
        super(StandardTransformer, self).__init__()

        self.num_blocks, self.tokenizer, self.latent_dim, self.seq_len, self.qk_dim, self.n_heads, self.num_mlp_layers, self.mlp_dimensions, self.activ_func, self.v_dim, self.causal_mask= num_blocks, tokenizer, latent_dim, seq_len, qk_dim, n_heads, num_mlp_layers, mlp_dimensions, activ_func, v_dim, causal_mask # Surely there is a better way to unpack these args. I feel like a total slopper
        
        self.vocab_size = tokenizer.vocab_size

        self.embed = EmbeddingLayer(self.vocab_size, self.latent_dim)
        self.blocks = ModuleList([
            TransformerBlock(latent_dim, seq_len, qk_dim, n_heads, num_mlp_layers, mlp_dimensions, activ_func=activ_func, v_dim=v_dim, causal_mask=causal_mask) for l in range(self.num_blocks)
        ]) 
        print(self.num_blocks)
        print(self.blocks)
        self.unembed = UnembeddingLayer(self.vocab_size, self.latent_dim, tied_weight=self.embed.E)

    def forward(self, x: str):
        x = self.embed(self.tokenizer.tokenize(x))
        for l in range(self.num_blocks): 
            x = self.blocks[l](x)
        x = self.unembed(x)
        logits = softmax(x, dim=-1)
        return logits

"""
======== BASIC TESTS ========   

Just running the components on
simple inputs to make sure there
are no immediate errors.

"""

def run_basic_tests():
    # TOKENIZER TEST
    s="abcd hello"
    toy_tokenizer = ToyTokenizer()
    ids = toy_tokenizer.tokenize(s)
    # print(ids)


    # SOFTMAX TEST
    x = tls.rand(4,2,3)
    y_0 = softmax(x, dim=-1, masked=True, mask_dim=-2)
    # print("Softmax test: ", y_0)
    
    # LAYERNORM TEST
    x = tls.rand(3,3,4)
    LN = LayerNorm()
    # print(x)
    y_layernorm = LN(x)
    # print(y_layernorm)
    # print(tls.sum(y_layernorm, dim=-1))

    x = tls.ones(2,2)
    layer = Linear(2, 2)
    y_1 = layer(x)
    # print("Linear test: ", y_1)
    # print("MLP test: ", y_2)
    num_mlp_layers = 4
    mlp_dimensions = [8,11, 17, 16, 8]
    latent_dim = 8
    seq_len = 10
    n_heads = 4
    qk_dim = 4
    batch_size = 2

    x = tls.rand(batch_size, seq_len, latent_dim)

    mlp = MLP(num_mlp_layers, mlp_dimensions)
    y_2 = mlp(x)
 
    mha = MultiHeadAttention(latent_dim, seq_len, qk_dim, n_heads)
    
    y_3 = mha(x)
    # print("MHA test: ", y_3)

    # RuntimeError: einsum(): subscript a has size 8 for operand 1 which does not broadcast with previously seen 2 
    # so 
    transformer_block=TransformerBlock(latent_dim, seq_len, qk_dim, n_heads, num_mlp_layers, mlp_dimensions)
    y_4 = transformer_block(x)


    num_blocks = 3 
    transformer = StandardTransformer(num_blocks, toy_tokenizer, latent_dim, seq_len, qk_dim, n_heads, num_mlp_layers, mlp_dimensions)
    s="yooo"
    logits = transformer(s)
    # print(logits)

    print("Nice job, no errors!")


if __name__ == "__main__":
    run_basic_tests()
