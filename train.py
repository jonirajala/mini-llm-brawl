import numpy as np
import torch
import tiktoken
import os
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import *
import json
import argparse

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(model_name):
    model_classes = {
        "gpt": GPT,
        "mistral": Mistral,
        "llama": LLama,
        "gemma": Gemma,
        "baseline_transformer": Transformer,
    }
    return model_classes.get(model_name, None)

class DataLoader:
    def __init__(self, data, batch_size, block_size):
        self.data = data

        self.batch_size = batch_size
        self.block_size = block_size
        self.pos = 0
    
    def get_batch(self):
        B, T = self.batch_size, self.block_size
        batch = self.data[self.pos:self.pos+B*T+1]
        x = torch.tensor(batch[:-1], dtype=torch.long).reshape(B, T).to(device)
        y = torch.tensor(batch[1:], dtype=torch.long).reshape(B, T).to(device)
        self.pos += B * T

        if self.pos + (B*T+1) > len(self.data):
            self.pos = 0

        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a specific model.")
    parser.add_argument("model_name", nargs='?', help="Name of the model to load")
    args = parser.parse_args()
    model_name = args.model_name


    enc = tiktoken.get_encoding("gpt2")
    device = "mps"

    class Config:
        emb_dim = 512
        vocab_size = enc.n_vocab
        n_layers = 8
        n_head = 8
        block_size = 30
        batch_size = 32
        iters = 100
        dropout = 0.1
        window_size = block_size // 2
        n_groups = 8

    config = Config()

    train_data = np.memmap(os.path.join("data", 'shakespare_train.bin'), dtype=np.uint16, mode='r')
    train_data = np.array(train_data)


    models = ["gpt", "llama", "mistral", "baseline_transformer", "gemma"]
    if model_name:
        assert model_name in models, f"Model {model_name} not found"
        models = [model_name]

    for model_name in models:
        model = get_model(model_name)
        model = model(config).to(device)

        print(f"Model: {model_name:<10} | Params: {count_parameters(model):>10,}")

        optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        trainloader = DataLoader(train_data, config.batch_size, config.block_size)

        losses = []
        model.train()
        pbar = tqdm(range(config.iters), desc="Training Progress")
        for i in pbar:
            x, y = trainloader.get_batch()
            model.zero_grad()
            out, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"train_loss": loss.item()})
            losses.append(loss.item())


        with open(f'losses_{model_name}.json', 'w') as f:
            json.dump(losses, f)

        # model.eval()
        # gen_text = model.generate("I am ").detach().cpu().numpy()
        # print(gen_text)
        # gen_text = enc.decode(gen_text)
        # print(gen_text)

        # plt.plot(range(config.iters), losses, label="Training Loss")
        # plt.legend()
        # plt.show()
