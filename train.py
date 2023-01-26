import os
import time
from config import Config
import torch
from network import BigramLanguageModel
import keyboard

torch.manual_seed(1347)

with open('c_sharp_data.txt', 'r', encoding='utf-8-sig') as f:
    text = f.read()

enc = Config.enc
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
    x = torch.stack([data[i:i+Config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+Config.block_size+1] for i in ix])
    x, y = x.to(Config.device), y.to(Config.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(Config.eval_iters)
        for k in range(Config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

stop_trainig = False #solo para modo debug, para parar el entrenamiento.
def stop_trainig_loop():
    global stop_trainig
    print("Stoping....")
    stop_trainig = True

# Train and test splits
print("Reading, encoding and splitting data.")
data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print("Preparing Neural Network")
model = BigramLanguageModel()
m = model.to(Config.device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

#load model weights if exists
model_to_load = Config.best_model_name if Config.use_best_model==True else Config.model_name
path = os.path.join('experiments', Config.experiment)
if not os.path.exists(path):
    os.makedirs(path)
model_to_load = os.path.join(path, model_to_load)
best_loss = float("inf")
val_losess = []
train_losess = []
epoch_init = 0
if os.path.exists(model_to_load):
    checkpoint = torch.load(model_to_load, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_init = checkpoint['epoch']
    val_losess = checkpoint['val_losses']
    train_losess = checkpoint['train_losses']
    best_loss = min(val_losess)
    print(f"loaded model from {model_to_load}. Trained for {epoch_init} epochs with best loss of {best_loss}")


if epoch_init < Config.max_iters: # continue training
    keyboard.add_hotkey('ctrl + s', callback=stop_trainig_loop, args=())
    start_time = time.time()
    for iter in range(epoch_init+1, Config.max_iters):        
        # every once in a while evaluate the loss on train and val sets
        if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            print(f"step {iter}: elapsed {elapsed:.4f} segs, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            val_losess.append(losses['val'])
            train_losess.append(losses['train'])
            if losses['val'] < best_loss:
                best_loss = losses['val']
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': iter,
                            'val_losses': val_losess,
                            'train_losses': train_losess}, 
                            os.path.join(path, Config.best_model_name))
            start_time = time.time()
        # if iter % 20 == 0:
        #     elapsed = time.time() - start_time
        #     print(f"step {iter}: elapsed {elapsed:.4f} segs")
        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        epoch_init +=1           
        if stop_trainig:
            break
    torch.save({'epoch': epoch_init,
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),            
                'val_losses': val_losess,
                'train_losses': train_losess}, 
                os.path.join(path, Config.model_name))
print("Done....")
exit()