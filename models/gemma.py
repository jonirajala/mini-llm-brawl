"""
https://arxiv.org/pdf/2403.08295

quite similar to llama

differences to baseline transfomer

Multi-Query Attention (Shazeer, 2019). Notably, the 7B model uses multi-head attention
while the 2B checkpoints use multi-query attention (with 𝑛𝑢𝑚_𝑘𝑣_ℎ𝑒𝑎𝑑𝑠 = 1), based on ablations
that showed that multi-query attention works well
at small scales (Shazeer, 2019).
RoPE Embeddings (Su et al., 2021). Rather than
using absolute positional embeddings, we use rotary positional embeddings in each layer; we also
share embeddings across our inputs and outputs
to reduce model size.
GeGLU Activations (Shazeer, 2020). The standard ReLU non-linearity is replaced by the approximated version of the GeGLU activation function.
RMSNorm. We normalize the input of each transformer sub-layer, the attention layer and the feedforward layer, with RMSNorm (Zhang and Sennrich, 2019) to stabilize the training.

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


class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def geglu(self, x):
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x):
        return self.geglu(x)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.emb_dim) 
        self.fc2 = nn.Linear(config.emb_dim // 2, config.emb_dim)
        self.geglu = GEGLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.geglu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class MultiQueryAttention(nn.Module):
    def __init__(self, config):
        super(MultiQueryAttention, self).__init__()
        self.emb_dim = config.emb_dim
        self.n_head = config.n_head
        self.head_dim = config.emb_dim // config.n_head

        assert self.head_dim * config.n_head == self.emb_dim, "emb_dim must be divisible by n_head"

        self.q_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.k_proj = nn.Linear(config.emb_dim, self.head_dim)  # Shared key
        self.v_proj = nn.Linear(config.emb_dim, self.head_dim)  # Shared value
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)

        self.pos_emb = RotaryPositionalEmbeddings(config.emb_dim // config.n_head, config.block_size)


    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim)  # Shared key
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim)  # Shared value

        q = q.permute(0, 2, 1, 3)  # (batch_size, n_head, seq_len, head_dim)
        k = k.permute(0, 2, 1, 3)  # (batch_size, 1, seq_len, head_dim)
        v = v.permute(0, 2, 1, 3)  # (batch_size, 1, seq_len, head_dim)

        q, k = self.pos_emb(q), self.pos_emb(k)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, emb_dim)
        
        # Output projection
        output = self.out_proj(attn_output)

        return output

        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rn1 = RMSNorm(config.emb_dim)
        self.rn2 = RMSNorm(config.emb_dim)

        self.attn = MultiQueryAttention(config)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.rn1(x)))
        x = x + self.dropout(self.mlp(self.rn2(x)))
        return x
        

class Gemma(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inp_emb = nn.Embedding(config.vocab_size, config.emb_dim)

        self.dropout = nn.Dropout(config.dropout)
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
