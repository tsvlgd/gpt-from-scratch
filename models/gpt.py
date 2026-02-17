import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int 
    batch_size: int = 64
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6 
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True


class Head(nn.Module):
    """" One head of self-attention """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."
        self.head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, config.head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * self.head_size **-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiheadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)]) # output of each head is (B,T,head_size)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate head outputs along the embedding dimension
        out = self.dropout(self.proj(out))
        return out


class MLP(nn.Module):
    """ A simple feed forward network """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias), 
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiheadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd) # layer norm 2 
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.mlp(self.ln2(x))
        return x 


class GPT(nn.Module):
    """ GPT architecture """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Linear layers Noraml distribution
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Embedding layer
    
    
    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()

        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        pos = torch.arange(0, T, device=device)

        tok_emb = self.transformer.wte(idx)        # (B,T,C)
        pos_emb = self.transformer.wpe(pos)       # (T,C)

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # get last token and apply temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx






