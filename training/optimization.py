import tools as tls
from tools import Tensor


"""
Loss functions
- take in tensor of correct shape
    - e.g [batch size, seq_len, logits] 
- take in tokenized correct datapoints
- calculate a single number and return it
"""

# ok I need to rewrite the model to use tensor token labels instead of 

def cross_entropy_loss(y_pred: Tensor, labels: Tensor):
    """
    Tensor is shape [batch_size, seq_len, vocab_size]
    take in Tensor of logits and tokenized class labels
    create tensor mask, ignoring if x
    zero out everything except the predicted class
    """
    tensor_mask = tls.zeros_like(y_pred)


    



"""
Optimizers
- init
-step 
- zero grad
"""

