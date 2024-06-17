"""
BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding

One goal to make a llm which can be easily transferlearned
model that can take all tokens into account -> predict tokens in middle of sentence 


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
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, random_split




class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.emb_dim) 
        self.fc2 = nn.Linear(config.emb_dim, config.emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class BidirectionalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_head == 0
        self.fc_in = nn.Linear(config.emb_dim, config.emb_dim * 3)
        self.fc_out = nn.Linear(config.emb_dim, config.emb_dim)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.emb_dim = config.emb_dim
        self.n_head = config.n_head
    
    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        q, k, v = self.fc_in(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # Shape (B, 1, 1, T) for broadcasting
            att = att.masked_fill(attention_mask == 0, float('-inf'))

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
        self.attn = BidirectionalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
        

class Bert(nn.Module):
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

    def forward(self, x, attn_mask=None):
        device = x.device  # Ensure device is defined
        batch, seq_len = x.shape
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # Masking
        if self.training:
            x, masked_labels = create_masked_lm_predictions(x, self.config.vocab_size, self.config.mask_token_id, self.config.pad_token_id)
        else:
            masked_labels = None

        x = self.dropout(self.inp_emb(x) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln(x)
        logits = self.fc1(x)

        loss = None        
        if self.training:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), masked_labels.view(-1), ignore_index=self.config.pad_token_id)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, inp, temperature=1.0, top_k=None):
        inp = tokenizer(
            inp,
            # padding='max_length',
            # truncation=True,
            # max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        inp = inp['input_ids'].reshape(1, -1)
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

def create_masked_lm_predictions(tokens, vocab_size, mask_token_id, pad_token_id):
    labels = tokens.clone()
    probability_matrix = torch.full(labels.shape, 0.15).to(device)
    special_tokens_mask = (tokens == pad_token_id)
    
    # Avoid masking special tokens (like pad tokens)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    labels[~masked_indices] = -1  # Only compute loss for masked tokens
    
    # 80% of the time, replace masked input tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(device)).bool() & masked_indices
    tokens[indices_replaced] = mask_token_id

    # 10% of the time, replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5).to(device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long).to(device)
    tokens[indices_random] = random_words[indices_random]

    # The rest 10% of the time, keep the masked input tokens unchanged

    return tokens, labels



# enc = tiktoken.get_encoding("gpt2")
device = "mps"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

class Config:
    emb_dim = 512
    vocab_size = tokenizer.vocab_size
    n_layers = 8
    n_head = 8
    block_size = 30
    batch_size = 32
    iters = 2000
    dropout = 0.1
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id


config = Config()


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length):
        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = self.split_into_chunks(text)

    def split_into_chunks(self, text):
        words = text.split()
        chunks = []
        chunk = []
        current_length = 0
        for word in words:
            word_length = len(self.tokenizer.tokenize(word))
            if current_length + word_length > self.max_length - 2:  # account for special tokens
                chunks.append(" ".join(chunk))
                chunk = [word]
                current_length = word_length
            else:
                chunk.append(word)
                current_length += word_length
        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        text = self.chunks[idx]
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded['input_ids'].squeeze().to(device), encoded['attention_mask'].squeeze().to(device)


# Load the corpus
with open("data/raw_shakespare.txt", 'r', encoding='utf-8') as f:
    corpus = f.read()

# Define the maximum sequence length and batch size
max_length = 128
batch_size = 16
iters = 500

# Create the dataset
dataset = TextDataset(corpus, tokenizer, max_length)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for each set
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Bert(config)
model.to(device)

optim = optim.AdamW(model.parameters(), lr=3e-4)


losses = []
model.train()
for i in range(iters):
    x, att_mask = next(iter(train_dataloader))
    model.zero_grad()
    out, loss = model(x, att_mask)
    loss.backward()
    optim.step()

    losses.append(loss.item())    
    print(f"iter {i}, loss: {loss.item()}")

model.eval()
inp = "Hello I am"
out = model.generate(inp)
print(tokenizer.decode(out))

plt.plot(range(iters), losses, label="Training Loss")
plt.legend()
plt.show()
