"""
https://arxiv.org/pdf/2401.04088

same architecture as mistral but uses "mixture-of-experts" layer instead of the mlp layer in the block

Mixtral is based on a transformer architecture [31] and uses the same
modifications as described in [18], with the notable exceptions that Mixtral supports a fully dense context length of 32k tokens, and the feedforward blocks are replaced by Mixture-of-Expert layers (Section 2.1).
The model architecture parameters are summarized in Table 1.

Mixtral is a sparse mixture-of-experts network. It is a decoder-only model where the feedforward
block picks from a set of 8 distinct groups of parameters. At every layer, for every token, a router
network chooses two of these groups (the “experts”) to process the token and combine their output
additively. This technique increases the number of parameters of a model while controlling cost and
latency, as the model only uses a fraction of the total set of parameters per token.

"""

from torch import nn
import torch
import math
from torch.nn import functional as F
from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings


class Config:
    emb_dim = 384
    n_layers = 6
    n_head = 6
    n_kv_heads = 6

    num_experts = 5
    top_k = 2

    def __init__(self, config):
        self.config = config
        if config.param_count == 50:
            emb_dim = 384
        elif config.param_count == 75:
            emb_dim = 384

        setattr(self, "emb_dim", emb_dim)

    def __getattr__(self, name):
        # Return attributes from self.config if not found in self
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'Config_50' object has no attribute '{name}'")


# Swiglu
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


class MoeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.emb_dim, config.num_experts, bias=False)
        self.config = config

    def forward(self, x):
        gate_logits = self.gate(x)


        top_k_logits, top_k_indices = gate_logits.topk(self.config.top_k, dim=-1)

        zeros = torch.full_like(gate_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
        gating_output = F.softmax(sparse_logits, dim=-1)

        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (top_k_indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class SlidingWindowSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_head == 0

        self.emb_dim = config.emb_dim
        self.n_heads_q = config.n_head  # Number of query heads
        self.n_kv_heads = config.n_kv_heads  # Number of key/value heads
        self.n_rep = (
            self.n_heads_q // self.n_kv_heads
        )  # Repetitions to match query heads
        self.head_dim = config.emb_dim // self.n_heads_q

        self.Wq = nn.Linear(config.emb_dim, self.n_heads_q * self.head_dim, bias=False)
        self.Wk = nn.Linear(config.emb_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(config.emb_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.n_heads_q * self.head_dim, config.emb_dim, bias=False)
        self.register_buffer(
            "bias",
            self._create_sliding_window_bias(config.block_size, config.window_size),
        )
        self.pos_emb = RotaryPositionalEmbeddings(self.head_dim, config.block_size)

    def _create_sliding_window_bias(self, block_size, window_size):
        bias = torch.zeros(block_size, block_size)
        for i in range(block_size):
            start = max(0, i - window_size)
            end = i + 1
            bias[i, start:end] = 1
        return bias.view(1, 1, block_size, block_size)

    def repeat_heads(self, x, n_rep):
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return (
                x[:, :, :, None, :]
                .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
            )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        assert dim == self.emb_dim, "dim must be equal to self.emb_dim"

        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = self.pos_emb(xq)
        xk = self.pos_emb(xk)

        keys = self.repeat_heads(xk, self.n_rep)
        values = self.repeat_heads(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

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
        self.attn = SlidingWindowSelfAttention(config)
        # self.mlp = MLP(config)
        self.moe = MoeLayer(config)

    def forward(self, x):
        x = x + self.attn(self.rn1(x))
        x = x + self.moe(self.rn2(x))
        return x


class Mixtral(nn.Module):
    def __init__(self, glob_config):
        super().__init__()
        config = Config(glob_config)
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
