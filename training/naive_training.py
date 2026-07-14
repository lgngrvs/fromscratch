import tools as tls
from architecture.layers import ToyTokenizer, StandardTransformer, MLP, Tokenizer, MultiHeadAttention
from training.optimization import SGD, Optimizer, cross_entropy_loss, accuracy
import plotext as pltxt
"""
TODO BEFORE TRAINING:
[x] Implement train script
    [x] Implement boilerplate
    [x] Implement loss function
    [x] Implement an optimizer
[x] Implement dataset generation
Improvements:
    [x] Add validation set
    [ ] Implement positional encoding
    [x] Implement ignoring tokens 
    [x] Implement loss graphing
    [ ] Add training for other models
"""


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


def plot_train_val_curves(loss_history: list[float], val_history: list[float], val_steps: list[int]):
    """
    Plots train and validation curves using loss history and validation history.
    """
    pltxt.subplots(2,1)
    pltxt.theme("pro")
    pltxt.subplot(1,1)
    pltxt.title("Loss")
    pltxt.scatter(loss_history)

    pltxt.subplot(2,1)
    pltxt.title("Validation Accuracy")
    pltxt.hline(1.0)
    pltxt.scatter(val_steps, val_history)
    pltxt.show()

prompts=["aaaa", "abcd", "bcde" "fghi", "JKL", "wahoo!" "ahsoeh"]

def sample_completions_greedy(model, prompts: list[str], max_generation_length: int):
    """
    Returns completions run by greedy-decoding the model 
    """
    for prompt in prompts:
        out = model.generate_rollout(prompt, max_generation_length)
        print(out)

def train(model, dataset, labels, val_dataset, val_labels, batch_size: int, epochs: int, optimizer, loss_function=cross_entropy_loss, print_loss_freq: int = 5, val_freq: int = 50, using_forward_with_logits: bool=False, tokenizer:Tokenizer = None):
    print("Training started.")
    if tokenizer is None: 
        tokenizer = model.tokenizer # Try to infer tokenizer from model
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

            if using_forward_with_logits:
                logits = model.forward_with_logits(batch) # Wrapper added for non-transformer Modules
            else: 
                logits = model(batch)
            loss = loss_function(logits, batch_labels, ignore_token_id=tokenizer.pad_token_id)
            loss_history.append(loss.item())
            loss.backward() # computes the gradient of the parameters wrt the loss. You can access the gradient using Tensor.grad. This is what we will be doing.
            optimizer.step() # The optimizer already has the model loaded into it, and will access the gradients that way directly. Thus, all we need to do is tell the optimizer to take a step in the direction of that gradient we've accumulated.
            if current_train_step % print_loss_freq == 0:
                print(f"Loss at step {current_train_step}: {loss.item()}")
            if current_train_step % val_freq == 0: 
                with tls.no_grad():
                    if using_forward_with_logits:
                        val_acc = accuracy(model.forward_with_logits(val_dataset), val_labels, ignore_token_id=tokenizer.pad_token_id)
                    else:
                        val_acc = accuracy(model(val_dataset), val_labels, ignore_token_id=tokenizer.pad_token_id)
                    val_history.append(val_acc.item())
                    val_steps.append(current_train_step)
                    print(f"Validation accuracy at step {current_train_step}: {val_acc.item()}")
            current_train_step+=1
    plot_train_val_curves(loss_history, val_history, val_steps)
            

if __name__ == "__main__":
    """
    Runs the naive training script.
    """


    """
    TRAINING PARAMETERS
    """
    BATCH_SIZE=16
    EPOCHS=4 # will be overridden below
    LR=1e-4
    TOKENIZER = ToyTokenizer()


    """
    TRAIN A TRANSFORMER
    """
    EPOCHS=2


    NUM_BLOCKS = 2 # VERY SIMPLE
    LATENT_DIM = 8
    SEQ_LEN = 27
    N_HEADS = 2
    QK_DIM = 4
    NUM_MLP_LAYERS = 2
    MLP_DIMENSIONS = [LATENT_DIM, LATENT_DIM*2, LATENT_DIM]

    MODEL = StandardTransformer(NUM_BLOCKS, TOKENIZER, LATENT_DIM, SEQ_LEN, QK_DIM, N_HEADS, NUM_MLP_LAYERS, MLP_DIMENSIONS)
    DATASET_RAW=generate_alphabet_dataset(int(4096), length=SEQ_LEN)
    DATASET, LABELS = TOKENIZER.batch_tokenize_and_pad(DATASET_RAW, max_seq_len=SEQ_LEN)
    val_dataset=generate_alphabet_dataset(int(32), length=SEQ_LEN)
    VAL_DATASET, VAL_LABELS = TOKENIZER.batch_tokenize_and_pad(val_dataset, max_seq_len=SEQ_LEN)
    OPTIMIZER=SGD(MODEL, LR)

    train(MODEL, DATASET, LABELS, VAL_DATASET, VAL_LABELS, BATCH_SIZE, EPOCHS, OPTIMIZER)


    """
    TRAIN AN MLP
    """
    EPOCHS=2

    VOCAB_SIZE=TOKENIZER.vocab_size
    print(VOCAB_SIZE)
    NUM_MLP_ONLY_LAYERS = 1
    MLP_ONLY_DIMENSIONS = [VOCAB_SIZE, VOCAB_SIZE] 
    
    MLP_MODEL = MLP(NUM_MLP_ONLY_LAYERS, MLP_ONLY_DIMENSIONS)
    MLP_OPTIMIZER=SGD(MLP_MODEL, LR)

    # You have to pre-one-hot-ify the dataset. Normally the embedding layer just does this for you instead of having to materialize the one-hot tensor, using just E[x] but we don't have that custom built.
    OH_VAL_DATASET=tls.one_hot(VAL_DATASET, num_classes=VOCAB_SIZE).float()
    OH_TRAIN_DATASET=tls.one_hot(DATASET, num_classes=VOCAB_SIZE).float()


    train(MLP_MODEL, OH_TRAIN_DATASET, LABELS, OH_VAL_DATASET, VAL_LABELS, BATCH_SIZE, EPOCHS, MLP_OPTIMIZER, using_forward_with_logits=True, tokenizer=TOKENIZER)


    # val_ds_responses=tls.argmax(MLP_MODEL.forward_with_logits(OH_VAL_DATASET), dim=-1)
    # print(VAL_LABELS)
    # print(val_ds_responses == VAL_LABELS)

    """
    TRAIN AN ATTENTION-ONLY SINGLE LAYER (lol cursed)
    """
    EPOCHS=6

    SINGLE_HEAD = MultiHeadAttention(58, SEQ_LEN, QK_DIM, N_HEADS)
    HEAD_OPTIMIZER = SGD(SINGLE_HEAD, LR)

    train(SINGLE_HEAD, OH_TRAIN_DATASET, LABELS, OH_VAL_DATASET, VAL_LABELS, BATCH_SIZE, EPOCHS, HEAD_OPTIMIZER, using_forward_with_logits=True, tokenizer=TOKENIZER)

