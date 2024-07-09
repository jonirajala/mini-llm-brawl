"""
https://arxiv.org/pdf/2310.11453
BitNet: Scaling 1-bit Transformers for Large Language Models
quantization-aware training for 1-bit large language models

BitNet uses the same layout as Transformers, stacking blocks of self-attention
and feed-forward networks. Compared with vanilla Transformer, BitNet uses BitLinear (Eq. 11)
instead of conventional matrix multiplication, which employs binarized (i.e., 1-bit) model weights.

Straight-through estimator. To train our 1-bit model, we employ the straight-through estimator
(STE)[BLC13] to approximate the gradient during backpropagation. This method bypasses the nondifferentiable functions, such as the Sign (Eq. 2) and Clip (Eq. 5) functions, during the backward pass.
STE allows gradients to flow through the network without being affected by these non-differentiable
functions, making it possible to train our quantized model.
"""

from torch import nn
import torch
import math
from torch.nn import functional as F


class STEQuantizeSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class STEQuantizeClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min_val, max_val):
        return torch.clamp(input, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

class BitLinear(nn.Module):
    def __init__(self, inp, out, before_activation=True):
        super(BitLinear, self).__init__()
        self.inp = inp
        self.out = out
        self.before_activation = before_activation
        self.weight = nn.Parameter(torch.Tensor(out, inp))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        w = self.weight
        n, m = w.shape
        
        # Weight quantization using STE
        alpha = w.mean()
        quant_w = STEQuantizeSign.apply(w - alpha)

        # Layer normalization of input
        x_norm = F.layer_norm(x, x.size()[1:])
        
        # Activation quantization parameters
        b = 8  # bit precision
        Qb = 2 ** (b - 1)
        gamma = torch.max(torch.abs(x))
        eps = 1e-5
        beta = torch.norm(w, p=1) / (n * m)
        scale = beta * gamma / Qb

        # Quantize activations using STE
        if self.before_activation:
            eta = torch.min(x_norm)
            quant_x = STEQuantizeClip.apply((x_norm - eta) * Qb / gamma, eps, Qb - eps)
        else:
            quant_x = STEQuantizeClip.apply(x_norm * Qb / gamma, -Qb + eps, Qb - eps)

        # Matrix multiplication with binarized weights and scaling
        y = F.linear(quant_x, quant_w) * scale
        return y
    
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = BitLinear(config.emb_dim, config.emb_dim, before_activation=True)
        self.fc2 = BitLinear(config.emb_dim, config.emb_dim, before_activation=False)
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
        self.emb_dim = config.emb_dim
        self.n_head = config.n_head

        self.fc_in = BitLinear(config.emb_dim, config.emb_dim * 3, before_activation=False)
        self.fc_out = BitLinear(config.emb_dim, config.emb_dim, before_activation=False)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.fc_in(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
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
        attn_output = self.attn(x)
        x = x + attn_output
        x = self.ln1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.ln2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class BitNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inp_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.positional_encoding = PositionalEncoding(config.emb_dim, config.block_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.fc_out = BitLinear(config.emb_dim, config.vocab_size, before_activation=False)
        self.ln = nn.LayerNorm(config.emb_dim)

        # self.inp_emb.weight = self.fc_out.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, x, y=None):
        x = self.inp_emb(x) * math.sqrt(self.inp_emb.embedding_dim)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
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
            50: [{"emb_dim": 432, "n_layers": 8, "n_head": 8}],
            75: [{"emb_dim": 576, "n_layers": 8, "n_head": 8}],
            100: [{"emb_dim": 672, "n_layers": 12, "n_head": 8}],
        }
        return param_configurations.get(params)
