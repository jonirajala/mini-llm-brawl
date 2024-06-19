"""
https://arxiv.org/pdf/2310.06825

like a llama with
grouped-query attention (GQA) [1], and sliding window attention (SWA)

GQA is a variant of multihead attention (MHA) that uses fewer write heads
    (key / value) than query heads.  GQA can be viewed as a generalization of
    multi-query attention (MQA), which uses a single write head. GQA and MQA give
    significant speedups over standard MHA in decoder layers, with minimal loss in
    accuracy. In the paper, GQA is shown to be more accurate than MQA, while still
    having a significant speedup over MHA.

"""

from torch import nn
import torch
import math
from torch.nn import functional as F
from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        scaled_hidden = int(2/3 * 4 * config.emb_dim)
        self.fc1 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc3 = nn.Linear(scaled_hidden, config.emb_dim, bias=False)
    
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1)
        hidden = hidden * x2
        return self.fc3(hidden)
    
class SlidingWindowSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_head == 0

        self.emb_dim = config.emb_dim
        self.n_heads_q = config.n_head  # Number of query heads
        self.n_kv_heads = config.n_kv_heads  # Number of key/value heads
        self.n_rep = self.n_heads_q // self.n_kv_heads  # Repetitions to match query heads

        self.head_dim = config.emb_dim // self.n_heads_q

        self.Wq = nn.Linear(config.emb_dim, self.n_heads_q * self.head_dim, bias=False)
        self.Wk = nn.Linear(config.emb_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(config.emb_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.n_heads_q * self.head_dim, config.emb_dim, bias=False)

        self.register_buffer("bias", self._create_sliding_window_bias(config.block_size, config.window_size))

        self.pos_emb = RotaryPositionalEmbeddings(self.head_dim, config.block_size)

    def _create_sliding_window_bias(self, block_size, window_size):
        bias = torch.zeros(block_size, block_size)
        for i in range(block_size):
            for j in range(max(0, i - window_size // 2), min(block_size, i + window_size // 2 + 1)):
                bias[i, j] = 1
        return bias.view(1, 1, block_size, block_size)

    def repeat_heads(self, x, n_rep):
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return (x[:, :, :, None, :]
                    .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                    .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
                    )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        assert dim == self.emb_dim, "dim must be equal to self.emb_dim"

        # Compute query, key, and value projections
        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        # Reshape and add positional embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = self.pos_emb(xq)
        xk = self.pos_emb(xk)

        # Repeat the heads of K and V to match the number of heads in Q
        keys = self.repeat_heads(xk, self.n_rep)
        values = self.repeat_heads(xv, self.n_rep)

        # Compute attention scores and apply attention mechanism
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.bias[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        context = torch.matmul(scores, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Apply output projection
        output = self.Wo(context)

        return output

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
        self.fc_out = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.rmsnorm = RMSNorm(config.emb_dim)
        # self.inp_emb.weight = self.fc_out.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, x, y=None):
        batch, seq_len = x.shape
        x = self.inp_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.rmsnorm(x)
        logits = self.fc_out(x)
        loss = None        
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        return logits, loss
    
    @torch.no_grad()
    def generate(self, inp, temperature=1.0, top_k=None):
        inp = inp.reshape(1, -1)
        for _ in range(self.config.block_size - inp.shape[1]):
            logits, _ = self.forward(inp)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            inp_next = torch.multinomial(probs, num_samples=1)
            inp = torch.cat((inp, inp_next), dim=1)
        return inp[0]