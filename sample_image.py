import os, pickle, argparse, torch
from model import GPTConfig, GPT          # same transformer, unchanged

p = argparse.ArgumentParser()
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--num", type=int, default=1, help="how many digits to draw")
args = p.parse_args()

device = 'cpu'
data_dir = os.path.join('data', 'mnist')

# meta.pkl gives the image geometry we need to fold the sequence back into a grid
meta = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
H, W, seq_len = meta['img_h'], meta['img_w'], meta['seq_len']

# rebuild the exact model from the checkpoint, then load the trained weights
ckpt = torch.load('ckpt.pt', map_location=device)
model = GPT(GPTConfig(**ckpt['config']))
model.load_state_dict(ckpt['model'])
model.eval()

for _ in range(args.num):
    # seed with the top-left pixel, which in MNIST is almost always background (0),
    # then let the model draw the remaining seq_len - 1 pixels.
    seed = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(seed, max_new_tokens=seq_len - 1, temperature=args.temperature)

    img = out[0, :seq_len].reshape(H, W)
    print()
    for row in img:
        print("".join("##" if v else ".." for v in row))
