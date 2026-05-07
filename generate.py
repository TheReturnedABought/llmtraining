import torch
from model import WorldModel
from config import ModelConfig
from data import encode, decode

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = ModelConfig()
model = WorldModel(cfg).to(device)
model.eval()


def generate(prompt, max_new=200):
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new):
            logits, _, _ = model(idx)
            logits = logits[:, -1] / 0.8
            probs = torch.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())


print(generate("Once upon a time "))