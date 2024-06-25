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

from torch import nn
import torch
import math
from torch.nn import functional as F
from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        scaled_hidden = int(2 / 3 * 4 * config.emb_dim)
        self.fc1 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc3 = nn.Linear(scaled_hidden, config.emb_dim, bias=False)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1)
        hidden = hidden * x2
        return self.fc3(hidden)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_head == 0
        self.emb_dim = config.emb_dim
        self.n_head = config.n_head
        self.head_dim = config.emb_dim // config.n_head

        self.Wq = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wk = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wv = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wo = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.pos_emb = RotaryPositionalEmbeddings(self.head_dim, config.block_size)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        assert dim == self.emb_dim, "dim must be equal to self.emb_dim"

        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        xq = xq.view(batch_size, seq_len, self.n_head, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_head, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_head, self.head_dim)

        xq = self.pos_emb(xq)
        xk = self.pos_emb(xk)

        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        context = torch.matmul(scores, values)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.Wo(context)
        return output


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rn1 = RMSNorm(config.emb_dim)
        self.rn2 = RMSNorm(config.emb_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.rn1(x))
        x = x + self.mlp(self.rn2(x))
        return x


class LLama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.inp_emb = nn.Embedding(config.vocab_size, config.emb_dim)
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
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, inp, temperature=1.0, top_k=None):
        inp = inp.reshape(1, -1)
        for _ in range(self.config.block_size - inp.shape[1]):
            logits, _ = self.forward(inp)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            inp_next = torch.multinomial(probs, num_samples=1)
            inp = torch.cat((inp, inp_next), dim=1)

        return inp[0]

    def get_param_conf(params):
        param_configurations = {
            50: [{"emb_dim": 384, "n_layers": 8, "n_head": 8}],
            75: [{"emb_dim": 512, "n_layers": 8, "n_head": 8}],
            100: [{"emb_dim": 576, "n_layers": 10, "n_head": 8}],
        }
        return param_configurations.get(params)
