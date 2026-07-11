import tools as tls
from tools import Module, Parameter, Tensor, ModuleList
from activations import ReLU

"""

MLP:
    
    Initialize weights (save them internally?)
    self.W
    self.B = 

    Forward pass
    self.

    Backward pass not needed


"""
# hi

class Linear(Module):
    def __init__(self, dim_1: int, dim_2: int):
        super(Linear, self).__init__()
        self.w = Parameter(Tensor(dim_1, dim_2))
        self.b = Parameter(Tensor(dim_2,))
        tls.init.kaiming_normal_(self.w)
        tls.init.zeros_(self.b)

    def forward(self, x: Tensor):
        out = tls.einsum("...a, ab -> ...b", x, self.w) + self.b
        return out

class MLP(Module):
    def __init__(self, num_layers: int, dimensions:list[int]):
        super(MLP, self).__init__()
        assert len(dimensions) == num_layers + 1, "Dimensions list must have length == num_layers + 1" 
        self.layers = ModuleList([Linear(dimensions[l], dimensions[l+1]) for l in range(num_layers)])
        self.n_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[l]
            x = ReLU(x)
        return x
         
        

x = tls.ones(2,2)
layer = Linear(2, 2)
y = layer(x)
print(y)

mlp = MLP(2, [2,3,2])

print("Nice job, no errors!")
