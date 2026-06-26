import os, pickle, numpy as np, torch
from model import GPTConfig, GPT          # model.py is in this folder -- the SAME transformer, unchanged

device = 'cpu'
batch_size = 64
data_dir = os.path.join('data', 'mnist')
max_iters = 1500
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

# meta.pkl gives vocab_size and seq_len
meta = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
vocab_size, seq_len = meta['vocab_size'], meta['seq_len']

# token streams written by prepare.py (uint16). long dtype for the embedding lookup.
train_data = torch.from_numpy(np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16).astype(np.int64))
val_data   = torch.from_numpy(np.fromfile(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16).astype(np.int64))

block_size = seq_len - 1                  # predict pixel t+1 from pixels 0..t
model = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size,
                      n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False))
# ... optimizer + training loop: same shape as your char-GPT ...

# data loading
def get_batch(split):
    # each example is ONE whole image, aligned so position 0 is always the top-left pixel.
    # (a random offset into the stream would make "position 0" a random pixel -- the trap.)
    data = train_data if split == 'train' else val_data
    images = data.view(-1, seq_len)            # (num_images, seq_len)
    ix = torch.randint(images.shape[0], (batch_size,))
    x = images[ix, :-1].contiguous()           # pixels 0..seq_len-2  -> .contiguous() for targets.view(-1)
    y = images[ix, 1:].contiguous()            # pixels 1..seq_len-1
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- Starting Training Loop ---")
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("--- Training Loop Finished ---")

# save a checkpoint so sample_image.py can rebuild the model and generate
checkpoint = {
    'model': model.state_dict(),
    'config': dict(block_size=block_size, vocab_size=vocab_size,
                   n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False),
}
torch.save(checkpoint, 'ckpt.pt')
print("saved checkpoint to ckpt.pt")