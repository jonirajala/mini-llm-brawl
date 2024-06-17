"""
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
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

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.emb_dim) 
        self.fc2 = nn.Linear(config.emb_dim, config.emb_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
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
    
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.fc_in(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

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
        self.pos_emb = nn.Embedding(config.block_size, config.emb_dim)

        self.dropout = nn.Dropout(config.dropout)
        self.config = config

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        self.fc1 = nn.Linear(config.emb_dim, config.vocab_size)

        self.ln = nn.LayerNorm(config.emb_dim)

        # self.inp_emb.weight = self.fc1.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, x, y=None):
        batch, seq_len = x.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        x = self.dropout(self.inp_emb(x) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.fc1(x)

        loss = None        
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss
    