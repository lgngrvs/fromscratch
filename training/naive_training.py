import tools as tls
from architecture.layers import ToyTokenizer, StandardTransformer
from training.optimization import SGD, Optimizer, cross_entropy_loss, accuracy
import plotext

"""
SYNTHETIC DATA
Trying to create a basic task that the model can
learn to know it can learn end-to-end.
Simple task: repeat the alphabet forwards starting
from an arbitrary point up to 20 characters

In the future would like to have a real data
structure for datasets but for now that's not desirable
"""

def generate_alphabet_dataset(n_datapoints: int, length: int = 26):
    """
    Generates a dataset of strings which is just the alphabet 
    with an integer offset. Testing if the transformer can 
    learn a very basic deterministic mapping, just to
    prove it can learn at all.
   """
    string = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    dataset = []
    offsets=tls.randint(0,25, (n_datapoints,)) # tensor filled with n_datapoints random offsets
    for offset in offsets:
        dataset.append(string[offset:offset+length])
    return dataset

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
[x] Implement train script
    [x] Implement boilerplate
    [x] Implement loss function
    [x] Implement an optimizer
[x] Implement dataset generation
Improvements:
    [ ] Add validation set
    [ ] Implement positional encoding
    [ ] Implement ignoring tokens 
    [ ] Implement loss graphing
    [ ] Add training for other models

"""


"""
MODEL PARAMETERS
"""
NUM_BLOCKS = 2 # VERY SIMPLE
LATENT_DIM = 8
SEQ_LEN = 27
N_HEADS = 2
QK_DIM = 4
NUM_MLP_LAYERS = 2
MLP_DIMENSIONS = [LATENT_DIM, LATENT_DIM*2, LATENT_DIM]
BATCH_SIZE=16
EPOCHS=3
LR=1e-4



"""
Good to also test out MLPs, Linear training, etc. 
You can just have an MLP learn the same task
"""


tokenizer = ToyTokenizer()
model = StandardTransformer(NUM_BLOCKS, tokenizer, LATENT_DIM, SEQ_LEN, QK_DIM, N_HEADS, NUM_MLP_LAYERS, MLP_DIMENSIONS)
DATASET_RAW=generate_alphabet_dataset(int(4096), length=SEQ_LEN)
DATASET, LABELS = tokenizer.batch_tokenize_and_pad(DATASET_RAW, max_seq_len=SEQ_LEN)
# print(DATASET)

val_dataset=generate_alphabet_dataset(int(32), length=SEQ_LEN)
VAL_DATASET, VAL_LABELS = tokenizer.batch_tokenize_and_pad(val_dataset, max_seq_len=SEQ_LEN)
optimizer=SGD(model, LR)

print("Initialization successful. Starting training.")



def train(model, dataset, labels, val_dataset, val_labels, batch_size: int, epochs: int, optimizer, loss_function=cross_entropy_loss, print_loss_freq: int = 5, val_freq: int = 50):
    print("Training started.")
    n_batches = -(len(dataset) // -batch_size) # divide and round up
    loss_history=[]
    val_history=[]
    val_steps=[]
    current_train_step=0
    for epoch in range(epochs):
        print("Beginning epoch", epoch)
        print(f"Running {n_batches} batches.")
        for batch_idx in range(n_batches):
            optimizer.zero_grad() # Zeros out the gradient at the beginning of the training step.
            if batch_idx == n_batches - 1:
                batch = dataset[batch_size * batch_idx:]
                batch_labels = labels[batch_size * batch_idx:]
            else: 
                batch = dataset[batch_size * batch_idx : batch_size * (batch_idx + 1)]
                batch_labels = labels[batch_size * batch_idx : batch_size * (batch_idx + 1)]
            logits = model(batch)
            loss = loss_function(logits, batch_labels, ignore_token_id=tokenizer.pad_token_id)
            loss_history.append(loss.item())
            loss.backward() # computes the gradient of the parameters wrt the loss. You can access the gradient using Tensor.grad. This is what we will be doing.
            optimizer.step() # The optimizer already has the model loaded into it, and will access the gradients that way directly. Thus, all we need to do is tell the optimizer to take a step in the direction of that gradient we've accumulated.
            if current_train_step % print_loss_freq == 0:
                print(f"Loss at step {current_train_step}: {loss.item()}")
            if current_train_step % val_freq == 0: 
                with tls.no_grad():
                    val_acc = accuracy(model(val_dataset), val_labels, ignore_token_id=tokenizer.pad_token_id)
                    val_history.append(val_acc.item())
                    val_steps.append(current_train_step)
                    print(f"Validation accuracy at step {current_train_step}: {val_acc.item()}")
            current_train_step+=1
    plotext.subplots(2,1)
    plotext.theme("pro")
    plotext.subplot(1,1)
    plotext.title("Loss")
    plotext.scatter(loss_history)

    plotext.subplot(2,1)
    plotext.title("Validation Accuracy")
    plotext.hline(1.0)
    plotext.scatter(val_steps, val_history)
    plotext.show()

train(model, DATASET, LABELS, VAL_DATASET, VAL_LABELS, BATCH_SIZE, EPOCHS, optimizer)

prompts=["aaaa", "abcd", "bcde" "fghi", "JKL", "wahoo!" "ahsoeh"]

for prompt in prompts:
    out = model.generate_rollout(prompt, 16)
    print(out)

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

