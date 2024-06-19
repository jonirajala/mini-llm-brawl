"""
https://arxiv.org/pdf/2310.06825

like a llama with
grouped-query attention (GQA) [1], and sliding window attention (SWA)

"""

from torch import nn
import torch
import math
from torch.nn import functional as F
from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize nn.Linear layers with specified input and output dimensions
        scaled_hidden = int(2/3 * 4 * config.emb_dim)
        self.fc1 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc3 = nn.Linear(scaled_hidden, config.emb_dim, bias=False)
    
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
    
class SlidingWindowSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.emb_dim % config.n_head == 0
        
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_head
        self.n_kv_heads = config.n_head
        self.head_dim = config.emb_dim // config.n_head
        self.emb_dim = config.emb_dim
        self.n_head = config.n_head
        self.window_size = config.window_size
        self.n_groups = config.n_groups 

        self.wq = nn.Linear(
                config.emb_dim,
                config.n_head * self.head_dim,
                bias=False
        )
        self.wk = nn.Linear(
            config.emb_dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            config.emb_dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            config.n_head * self.head_dim,
            config.emb_dim,
            bias=False
        )
        

        self.pos_emb = RotaryPositionalEmbeddings(self.head_dim, config.block_size)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        
        # Grouped Query Attention: reduce the number of unique keys and values
        G = self.n_groups
        k = k.view(B, T, G, -1).transpose(1, 2)  # (B, G, T, emb_dim // G)
        v = v.view(B, T, G, -1).transpose(1, 2)  # (B, G, T, emb_dim // G)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        q, k = self.pos_emb(q), self.pos_emb(k)

        attn_windows = []
        for i in range(0, T, self.window_size):
            q_i = q[:, :, i:i+self.window_size]
            k_i = k[:, :, i:i+self.window_size]
            v_i = v[:, :, i:i+self.window_size]

            att = (q_i @ k_i.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            y_i = att @ v_i

            attn_windows.append(y_i)

        y = torch.cat(attn_windows, dim=2)  # concatenate the windows along the sequence length
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.wo(y)

        return y

        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rn1 = RMSNorm(config.emb_dim)
        self.rn2 = RMSNorm(config.emb_dim)
        self.attn = SlidingWindowSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.rn1(x))
        x = x + self.mlp(self.rn2(x))
        return x
        

class Mistral(nn.Module):
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
