import tools as tls
from tools import Tensor, Module


"""
Loss functions
- take in tensor of correct shape
    - e.g [batch size, seq_len, logits] 
- take in tokenized correct datapoints
- calculate a single number and return it
"""

def worse_cross_entropy_loss(y_pred: Tensor, labels: Tensor, ignore_token_id: int=-100):
    """
    y_pred is shape [batch_size, seq_len, vocab_size]
    Labels are shape [batch_size, seq_len]
    with label at index $i$ taken from position $i+1$
    (since this is next-token prediction). 
    Take in Tensor of logits and tokenized class labels
    Create bool tensor mask, ignoring if token is pad token
    or other ignored position.
    Zero out everything except the predicted class
    """

    tensor_mask = tls.zeros_like(y_pred, dtype=torch.bool) # shape [batch_size, seq_len, vocab_size]. will be True at correct class_id at each position in each batch, false elsewhere
    # i can't figure out the slicing so i'm going to do it as a for lopo and then see what happens
    for batch_id in range(y_pred.shape[0]):
        for seq_pos in range(y_pred.shape[1]):
            class_id = labels[batch_id, seq_pos]
            tensor_mask[batch_id, seq_pos, class_id] = True
    loss = tls.sum(
        tls.log(y_pred * tensor_mask)
    ) 
    return loss
    """
    Kept as an artifact of my original implementation. 
    you can do this tensor slicing using advanced indexing,
    by choosing [B, 1], [1, S], and index all at once
    or you could use scatter()
    but the best way to do this is just cherrypick our values
    using torch.gather.
    """


def cross_entropy_loss(y_pred: Tensor, labels: Tensor, ignore_token_id: int=-100): # ignore token id has changed
    """
    Takes in [batch_size, seq_len, vocab_size],
    labels [batch_size, seq_len].
    Gather along vocab_size dimension
    out[i,j, 0] = in[i,j,index[i,j,0]]
    so we need to unsqueeze the labels to get
    that extra necessary dimension.
    """
    relevant_logits = y_pred.gather(-1, labels.unsqueeze(-1)).squeeze() # gathering along last dim; now the same shape as the labels after squeezing to remove singleton
    ignore_positions_mask = (labels == ignore_token_id) 
    relevant_logits = relevant_logits.masked_fill(ignore_positions_mask, 1.0) # fills with 1, and log(1)=0 so this will no longer contribute to the loss
    loss = -tls.sum(tls.log(relevant_logits)) 
    return loss


def accuracy(y_pred: Tensor, labels: Tensor, ignore_token_id: int=67): # ignore token id has changed
    y_pred_ids=tls.argmax(y_pred, dim=-1).flatten() # now same shape as a label since dim=-1 removed
    ignore_positions_mask = (labels == ignore_token_id).flatten()
    # incorrect_mask = (y_pred_ids == labels.flatten())
    # print("wrong acc:", tls.sum(incorrect_mask)/tls.numel(labels))
    kept_preds= y_pred_ids[ignore_positions_mask == False]
    #print("Predictions: ", kept_preds[0:30])
    kept_labels = labels.flatten()[ignore_positions_mask == False]
    #print("Labels: ", kept_labels[0:30])
    correct_mask = (kept_preds == kept_labels)
    acc = tls.sum(correct_mask)/tls.numel(kept_preds)
    return acc

"""
Optimizers need:
- init
- step 
- zero grad
"""

class Optimizer():
    """
    Base Optimizer class.
    """
    pass

class SGD(Optimizer):
    """
    Regular SGD.
    """
    def __init__(self, model: Module, lr: float):
        super(SGD, self).__init__() # doesn't currently do anything, but am adding for future ref
        self.lr = lr
        self.model = model

    def step(self):
        for param in self.model.parameters():
            param.data = param - self.lr * param.grad
    
    def zero_grad(self):
        for param in self.model.parameters():
            param.grad = tls.zeros_like(param, requires_grad=False)



if __name__=="main":
    import architecture.layers
    NUM_BLOCKS = 2 # VERY SIMPLE
    LATENT_DIM = 8
    SEQ_LEN = 20
    N_HEADS = 2
    QK_DIM = 4
    NUM_MLP_LAYERS = 2
    MLP_DIMENSIONS = [LATENT_DIM, LATENT_DIM*2, LATENT_DIM]
    BATCH_SIZE = 2

    tokenizer = architecture.layers.ToyTokenizer()
    transformer = architecture.layers.StandardTransformer(NUM_BLOCKS, tokenizer, LATENT_DIM, SEQ_LEN, QK_DIM, N_HEADS, NUM_MLP_LAYERS, MLP_DIMENSIONS)


    dataset, labels = tokenizer.batch_tokenize_and_pad(["hi", "wahoooo"], SEQ_LEN)
    logits = transformer(dataset)
    loss = cross_entropy_loss(logits, labels)
    import torch.autograd
    torch.autograd.set_detect_anomaly(True, check_nan=False)
    loss.backward()

    optim = SGD(transformer, 1e-3)
    optim.step()

    optim.zero_grad()
    optim.step()





















    



