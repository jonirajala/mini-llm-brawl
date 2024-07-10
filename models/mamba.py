"""
https://arxiv.org/pdf/2312.00752
https://github.com/johnma2006/mamba-minimal
Uses selective state spaces insted of attention

"""

from torch import nn
import torch
import math
from torch.nn import functional as F
from einops import rearrange, repeat, einsum
from torchtune.modules import RMSNorm

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_proj = nn.Linear(config.emb_dim, config.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            bias=False,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        A = repeat(torch.arange(1, config.d_state + 1), 'n -> d n', d=config.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.emb_dim, bias=False)
        self.config = config

    def forward(self, x):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.config.d_inner, self.config.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output
    
    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.config.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


class Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inp_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.fc_out = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.ln1 = RMSNorm(config.emb_dim)
        self.ln2 = RMSNorm(config.emb_dim)

        self.inp_emb.weight = self.fc_out.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, x, y=None):

        x = self.inp_emb(x)

        for block in self.blocks:
            x = x + block(self.ln2(x))
        
        x = self.ln1(x)

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
            50: [{"emb_dim": 128, "n_layers": 8, "n_head": 8, "d_inner": 128*2, "d_conv": 4, "dt_rank": 16, "d_state":16}],
            75: [{"emb_dim": 576, "n_layers": 8, "n_head": 8, "d_inner": 1000, "d_conv": 4, "dt_rank": 16, "d_state":16}],
            100: [{"emb_dim": 672, "n_layers": 12, "n_head": 8, "d_inner": 1000, "d_conv": 4, "dt_rank": 16, "d_state":16}],
        }
        return param_configurations.get(params)
