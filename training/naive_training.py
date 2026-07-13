import tools as tls
from architecture.layers import ToyTokenizer, StandardTransformer







"""
MODEL PARAMETERS
"""
NUM_BLOCKS = 2 # VERY SIMPLE
LATENT_DIM = 8
SEQ_LEN = 20
N_HEADS = 2
QK_DIM = 4
NUM_MLP_LAYERS = 2
MLP_DIMENSIONS = [LATENT_DIM, LATENT_DIM*2, LATENT_DIM]
BATCH_SIZE = 2


"""
Good to also test out MLPs, Linear training, etc. 
You can just have an MLP learn the same task
"""


tokenizer = ToyTokenizer()

model = StandardTransformer(NUM_BLOCKS, tokenizer, LATENT_DIM, SEQ_LEN, QK_DIM, N_HEADS, NUM_MLP_LAYERS, MLP_DIMENSIONS)

# hhhhh i forgot positional encoding! 

"""
SYNTHETIC DATA
Trying to create a basic task that the model can
learn to know it can learn end-to-end.
Simple task: repeat the alphabet forwards starting
from an arbitrary point up to 20 characters

In the future would like to have my own data
structures for this
"""

def generate_alphabet_datapoint(length):
    pass

def generate_alphabet_dataset(n_datapoints: int = 200, length: int = 20):
    pass
    

"""
TRAINING SETUP

Ok, so what we need is batch size,
a list of datapoints, (at some point
we can implement streaming to train
more at scale), then we need to run backwards()
with the logits
"""

"""
TODO BEFORE TRAINING:
[ ] Implement train script
    [x] Implement boilerplate
    [ ] Implement loss function
    [ ] Implement an optimizer
[ ] Implement dataset generation
[ ] Implement positional encoding
[ ] Add training for other models

"""



BATCH_SIZE=16
EPOCHS=3

DATASET=generate_alphabet_dataset()
optimizer=None # need to go implement an optimizer...
# optimizer should take in

"""
optimizer needs:
- init
- step
- zero grad (this is also typically acceptable 
to in the model, but not going to do that
for now)
"""

def train(model, dataset, batch_size, epochs, loss_function, optimizer):
    for epoch in range(epochs):
        n_batches = -(len(dataset) // batch_size) # divide and round up
        for batch_idx in range(n_batches):
            optimizer.zero_grad() # Zeros out the gradient at the beginning of the training step.
            if batch_idx == n_batches - 1:
                batch = dataset[batch_size * batch_idx:]
            else: 
                batch = dataset[batch_size * batch_idx : batch_size * (batch_idx + 1)]

            tok_batch = tokenizer.tokenize_batch(batch) # NEED TO BATCHIFY THE TOKENIZATION
            logits = model(tok_batch)
            loss = loss_function(tok_batch, logits)
            loss.backward() # computes the gradient of the parameters wrt the loss. You can access the gradient using Tensor.grad. This is what we will be doing.
            optimizer.step() # The optimizer already has the model loaded into it, and will access the gradients that way directly. Thus, all we need to do is tell the optimizer to take a step in the direction of that gradient we've accumulated.


"""
At some point will need to create a generate() function
with the model so that it runs correctly --
decoding etc. -- for rollouts and such but
for now this is what we need.

I'm tempted to create a model-independent
harness that just takes in logits, runs the model
to a specified length (and can autodetect stop
tokens) so that I don't have to duplicate code.
It's a little weird but I guess I get to
do this however I want! I will learn how
my own choices affect the construction of the
library, but this seems like a good way to
simplify things.
"""

