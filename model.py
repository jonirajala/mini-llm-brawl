import numpy as np
from torch import nn
import torch
import tiktoken
import os
import math
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.emb_dim) 
        self.fc2 = nn.Linear(config.emb_dim, config.emb_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_head == 0
        self.fc_in = nn.Linear(config.emb_dim, config.emb_dim*3)
        self.fc_out = nn.Linear(config.emb_dim, config.emb_dim)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

        self.emb_dim = config.emb_dim
        self.n_head = config.n_head
    
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.fc_in(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        y = self.fc_out(y)

        return y

        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
        

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inp_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.vocab_size, config.emb_dim)

        self.config = config

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        self.fc1 = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, x, y=None):
        batch, seq_len = x.shape
        pos = torch.arange(0, seq_len).repeat(batch, 1).to(device)

        x = self.inp_emb(x) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        logits = self.fc1(x)

        loss = None        
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, inp, temperature=1.0, top_k=None):
        inp = torch.tensor(enc.encode(inp)).to(device)
        inp = inp.reshape(1, -1)
        for _ in range(self.config.block_size-inp.shape[1]):
            logits, _ = self.forward(inp)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            inp_next = torch.multinomial(probs, num_samples=1)
            inp = torch.cat((inp, inp_next), dim=1)
        
        return inp[0]

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
    


enc = tiktoken.get_encoding("gpt2")
device = "mps"

class Config:
    def __init__(self):
        self.emb_dim = 256
        self.vocab_size = enc.n_vocab
        self.n_layers = 4
        self.n_head = 4
        self.block_size = 30
        self.batch_size = 32
        self.iters = 1000
config = Config()

train_data = np.memmap(os.path.join("data", 'shakespare_train.bin'), dtype=np.uint16, mode='r')

train_data = np.array(train_data)
print(train_data.shape)
trainloader = DataLoader(train_data, config.batch_size, config.block_size)
model = GPT(config)
model.to(device)

optim = optim.AdamW(model.parameters(), lr=0.001)

x, y = trainloader.get_batch()

losses = []

pbar = tqdm(range(config.iters), desc="Training Progress")

for i in pbar:
    x, y = trainloader.get_batch()
    model.zero_grad()
    out, loss = model(x, y)
    loss.backward()
    optim.step()
    pbar.set_postfix({"train_loss": loss.item()})
    losses.append(loss.item())



gen_text = model.generate("hei olen ").detach().cpu().numpy()
print(gen_text)
gen_text = enc.decode(gen_text)
print(gen_text)

plt.plot(range(config.iters), losses, label="Training Loss")
plt.legend()
plt.show()


