import torch
import tiktoken
class Config(object):    
    enc = tiktoken.get_encoding('gpt2')
    vocab_size = enc.n_vocab
    # hyperparameters
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 200000
    eval_interval = 200
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 100
    n_embd = 504
    n_head = 12
    n_layer = 6
    dropout = 0.2
    use_best_model = True
    model_name = "c_sharp.tar"
    best_model_name = "c_sharp_best.tar"
    experiment = '1'    