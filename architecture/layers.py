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
    Toy character-level tokenizer for testing.
    """
    def __init__(self):
        super(ToyTokenizer, self).__init__()
        # Letter position in allowed_letters also defines token id
        self.allowed_letters = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?")
        self.vocab_size=len(self.allowed_letters)+1 # not inluding pad token. I hope this doesn't cause problems! 
        self.pad_token_id = len(self.allowed_letters)
        print(self.pad_token_id)
        
    def _tokenize_str(self, s: str):
        """
        DEPRECATED BUT KEEPING AROUND UNTIL tokenize_and_pack
        is written, just in case it turns out to be useful
        and I decide to keep it.
        Code is duplicated in batch_tokenize_and_pad,
        with the advantage that we get to tell the user
        which datapoint their error is in.
        Probably should just be part of tokenize_and_pad
        If you want to stringify a single word, just stick
        the str in a 1-length list and run batch_tokenize.
        """
        split = list(s)
        assert set(split).issubset(set(self.allowed_letters)), "Toy Tokenizer only accepts letters, spaces, and .,!?." 

        # turn string into list of ids and then stick that into a vector 
        id_list=tls.tensor([ int(self.allowed_letters.index(letter)) for letter in split ])
        return id_list 

    def batch_tokenize_and_pad(self, batch_strs: list[str], max_seq_len: int):
        """
        Returns a tensor of shape [len(batch_strs), max_seq_len], with entries padded at the end.
        """
        # run through each str, tokenize, then concat
        batch_tensor = tls.zeros(len(batch_strs), max_seq_len, dtype=int)
        for str_idx, string in enumerate(batch_strs):
            split_str = list(string)
            assert set(split_str).issubset(set(self.allowed_letters)), f"Problem with datapoint {str_idx}: Toy Tokenizer only accepts letters, spaces, and .,!?." 
            assert len(string) <= max_seq_len, f"Datapoint {str_idx} exceeds maximum sequence length!"
            vector = tls.tensor([ int(self.allowed_letters.index(letter)) for letter in split_str ], dtype=int)
            batch_tensor[str_idx, 0:len(string)] = vector.unsqueeze(0) # insert vector in; reshape by adding dimension at front to broadcast correctly
            batch_tensor[str_idx, len(string):] = self.pad_token_id # pad the remainder

        labels = tls.cat((batch_tensor[:,1:], tls.full((len(batch_strs),1), self.pad_token_id, dtype=int)), dim=-1)
        return batch_tensor, labels # shape [len(batch_strs), max_seq_len]. will be padded
    
    def de_tokenize(self, batch_ids: Tensor):
        strs_list = []
        for seq_id in range(batch_ids.shape[0]):
            string=""
            for token_idx in range(len(batch_ids[seq_id, :])): 
                token = batch_ids[seq_id, token_idx]
                letter = list(self.allowed_letters)[token]
                string += letter
            strs_list.append(string)
        return strs_list


                
        

    def batch_tokenize_and_pack():
        """
        TODO
        """
        pass

class EmbeddingLayer(Module):
    def __init__(self, vocab_size: int, latent_dim: int):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size, self.latent_dim = vocab_size, latent_dim
        self.E = Parameter(Tensor(self.vocab_size, self.latent_dim))
        tls.init.kaiming_normal_(self.E)

    def forward(self, x: Tensor) -> Tensor:
        """
        Takes in a batch of [batch_size, max_seq_len] 
        (possibly padded but that makes no diff)
        Each entry is a class id.
        We want
        [batch_size, max_seq_len, latent_dim]
        I still don't really understand how
        advanced indexing works and would like to learn
        about it someday. Might be good
        to write a blog post about though that's probably
        not counterfactual good time spent.
        """
        return self.E[x] # just read the rows rather than doing one-hot multiplication (wasting matmletters 

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

        
"""
======== NONLINEARITIES ========
"""

def ReLU(x: Tensor):
    """
    Literally ReLU.
    """
    mask = x < 0
    x = x * mask
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
        x = x.masked_fill(inf_mask, -tls.inf) # not an in_place operation
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
            v_dim = -(-latent_dim // n_heads) # round up
            
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
        x = x + self.mhsa(x)
        x = self.ln_1(x)
        x = x + self.mlp(x)
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
        self.unembed = UnembeddingLayer(self.vocab_size, self.latent_dim, tied_weight=self.embed.E)

    def forward(self, x: Tensor):
        x = self.embed(x)
        for l in range(self.num_blocks): 
            x = self.blocks[l](x)
        x = self.unembed(x)
        logits = softmax(x, dim=-1)
        return logits

    def generate_rollout(self, prompt, max_generation_tokens):
        tok, _ = self.tokenizer.batch_tokenize_and_pad([prompt], self.seq_len)
        while tls.max(tok) == 57: # still padded (cursed setup)
            # print(tok)
            first_pad_position=tls.argmax(tok[0,:]) #this only works for tokenizers with max token id = pad token
            # will need better engineering for arbitrary pad ids
            logits=self.forward(tok)[0, first_pad_position-1, :]
            #print(logits)
            #print(logits.shape)
            generated_token = tls.argmax(logits)
            first_pad_position = tls.argmax(tok)
            # print(generated_token, first_pad_position)
            tok[0,first_pad_position]=generated_token
        # print(tok)
        return self.tokenizer.de_tokenize(tok)


"""
======== BASIC TESTS ========   

Just running the components on
simple inputs to make sure there
are no immediate errors.

"""

def run_basic_tests():
    # TOKENIZER TEST
    s="abcd hello!!"
    toy_tokenizer = ToyTokenizer()
    ids = toy_tokenizer._tokenize_str(s)
    # print(ids)
    try: 
        s = "hi*"
        ids = toy_tokenizer.tokenize_str(s)
        #print(ids)
        # should assertionerror since * is not in allowed tokens
    except:
        #print("Error correctly raised by tokenizer")
        pass


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
    seq_len = 20
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
    dataset = ["hi!!", "yo whats up", "wahooooooo"]
    tok_dataset, labels = toy_tokenizer.batch_tokenize_and_pad(dataset, seq_len)
    logits = transformer(tok_dataset)
    # print(logits)
    print(tok_dataset)
    print(labels)

    print("Nice job, no errors!")


if __name__ == "__main__":
    run_basic_tests()
