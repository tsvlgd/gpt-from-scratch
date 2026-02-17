import torch
import numpy as np
import pickle

from gpt_trainer.models.gpt import GPT, GPTConfig

# -------------------
# Load dataset
# -------------------

data_dir = "data/shakespeare_char"

train_data = np.memmap(f"{data_dir}/train.bin", dtype=np.uint16, mode='r')
val_data   = np.memmap(f"{data_dir}/val.bin", dtype=np.uint16, mode='r')

with open(f"{data_dir}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

vocab_size = meta["vocab_size"]

# -------------------
# Configs
# -------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size
)

model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -------------------
# Batch loader
# -------------------

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# -------------------
# Estimate loss
# -------------------

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -------------------
# Training loop
# -------------------

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete.")
