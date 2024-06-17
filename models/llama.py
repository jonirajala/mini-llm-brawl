"""
https://arxiv.org/pdf/2302.13971

'We train large transformers on a large quantity of textual data using a standard optimizer.'
'We tokenize the data with the bytepair encoding (BPE)'


Differences compared to basic transformer

Pre-normalization [GPT3]
- To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function, 

SwiGLU activation function [PaLM].
- We replace the ReLU non-linearity by the SwiGLU activation function

Rotary Embeddings [GPTNeo].
- We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by Su et al. (2021), at each layer of the network.


AdamW optimizer
- β1 = 0.9, β2 = 0.95.

We use a cosine learning rate schedule, such that the final learning rate is equal to 10% of the maximal learning rate. We use a weight decay of 0.1 and gradient clipping of 1.0

no dropout https://github.com/openlm-research/open_llama/issues/22

"""




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
from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Initialize nn.Linear layers with specified input and output dimensions
        scaled_hidden = int(2/3 * 4 * hidden_dim)
        self.fc1 = nn.Linear(input_dim, scaled_hidden)
        self.fc2 = nn.Linear(input_dim, scaled_hidden)
        self.fc3 = nn.Linear(scaled_hidden, input_dim)
    
    def forward(self, x):
        # Linear transformation with the first layer
        x1 = self.fc1(x)
        # Linear transformation with the second layer
        x2 = self.fc2(x)
        # Apply SiLU activation to the result of the first transformation
        hidden = F.silu(x1)
        # Element-wise multiplication of SiLU result and second transformation
        hidden = hidden * x2
        # Final linear transformation with the third layer
        return self.fc3(hidden)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.emb_dim) 
        self.fc2 = nn.Linear(config.emb_dim, config.emb_dim)
        self.swiglu = SwiGLU(config.emb_dim, config.emb_dim)

    def forward(self, x):
        x = self.swiglu(self.fc1(x))
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

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.emb_dim = config.emb_dim
        self.n_head = config.n_head

        self.pos_emb = RotaryPositionalEmbeddings(config.emb_dim // config.n_head, config.block_size)
    
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.fc_in(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        q, k = self.pos_emb(q), self.pos_emb(k)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        y = self.resid_dropout(self.fc_out(y))

        return y

        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rn1 = RMSNorm(config.emb_dim)
        self.rn2 = RMSNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.rn1(x))
        x = x + self.mlp(self.rn2(x))
        return x
        

class LLama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inp_emb = nn.Embedding(config.vocab_size, config.emb_dim)

        self.config = config

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        self.fc1 = nn.Linear(config.emb_dim, config.vocab_size)

        self.rmsnorm = RMSNorm(config.emb_dim)

        # self.inp_emb.weight = self.fc1.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, x, y=None):
        batch, seq_len = x.shape

        x = self.inp_emb(x)

        for block in self.blocks:
            x = block(x)

        x = self.rmsnorm(x)
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
    emb_dim = 512
    vocab_size = enc.n_vocab
    n_layers = 8
    n_head = 8
    block_size = 30
    batch_size = 32
    iters = 2000
    dropout = 0.1

config = Config()

train_data = np.memmap(os.path.join("data", 'shakespare_train.bin'), dtype=np.uint16, mode='r')

train_data = np.array(train_data)
print(train_data.shape)
trainloader = DataLoader(train_data, config.batch_size, config.block_size)
model = LLama(config)
model.to(device)

optim = optim.AdamW(model.parameters(), lr=3e-4)

x, y = trainloader.get_batch()

losses = []

# for i in range(50):
#     model.zero_grad()
#     out, loss = model(x, y)
#     loss.backward()
#     optim.step()
#     print(f"step {i}, loss: {loss.item()}")

pbar = tqdm(range(config.iters), desc="Training Progress")
for i in pbar:
    x, y = trainloader.get_batch()
    model.zero_grad()
    out, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    pbar.set_postfix({"train_loss": loss.item()})
    losses.append(loss.item())

model.eval()
gen_text = model.generate("I am ").detach().cpu().numpy()
print(gen_text)
gen_text = enc.decode(gen_text)
print(gen_text)

plt.plot(range(config.iters), losses, label="Training Loss")
plt.legend()
plt.show()
