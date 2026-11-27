import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True
    use_rope: bool = False

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size <= 0 or self.vocab_size <= 0:
            raise ValueError("block_size and vocab_size must be positive")
        if self.use_rope and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = config.use_rope
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
            )
            positions = torch.arange(config.block_size).float()
            frequencies = torch.outer(positions, inv_freq)
            self.register_buffer("rope_cos", frequencies.cos(), persistent=False)
            self.register_buffer("rope_sin", frequencies.sin(), persistent=False)

    @staticmethod
    def _apply_rope(x, cos, sin):
        half_dim = x.size(-1) // 2
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    def forward(self, x):
        batch_size, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head

        q = q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        if self.use_rope:
            cos = self.rope_cos[:seq_len].to(dtype=q.dtype)
            sin = self.rope_sin[:seq_len].to(dtype=q.dtype)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": (
                    None
                    if config.use_rope
                    else nn.Embedding(config.block_size, config.n_embd)
                ),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)

        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block size {self.config.block_size}"
            )

        x = self.transformer["wte"](idx)
        if not self.config.use_rope:
            positions = torch.arange(seq_len, device=idx.device)
            x = x + self.transformer["wpe"](positions)
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        loss = None
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
        else:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay, no_decay = [], []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.dim() >= 2 else no_decay).append(parameter)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=betas,
            fused=device_type == "cuda",
        )

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
