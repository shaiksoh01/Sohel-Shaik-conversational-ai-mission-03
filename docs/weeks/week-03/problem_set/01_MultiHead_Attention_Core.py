"""
Mission 03 - Part 1: Scaled Dot-Product & Multi-Head Self-Attention from Scratch
Deliverable: 01_MultiHead_Attention_Core.py
Starship Operational Unit: USSE Noosphere Bridge Crew (Squadron Turing)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True):
        batch_size, seq_len, _ = x.shape
        q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(context), attn_weights


def run_variance_preservation_test():
    torch.manual_seed(42)
    batch_size, num_heads, seq_len, head_dim = 32, 8, 256, 128
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    unscaled_logits = torch.matmul(q, k.transpose(-2, -1))
    scaled_logits = unscaled_logits / math.sqrt(head_dim)

    print("=" * 72)
    print("PART 1: MULTI-HEAD SELF-ATTENTION & VARIANCE PRESERVATION TEST")
    print("=" * 72)
    print(f"Input Q Var   : {q.var().item():.4f} (Target ~ 1.0000)")
    print(f"Input K Var   : {k.var().item():.4f} (Target ~ 1.0000)")
    print(f"Unscaled Var  : {unscaled_logits.var().item():.4f} (Expected ~ d_k = {head_dim})")
    print(f"Scaled Var    : {scaled_logits.var().item():.4f} (Target ~ 1.0000)")
    assert abs(scaled_logits.var().item() - 1.0) < 0.05

    x = torch.randn(2, 16, 512)
    mha = MultiHeadSelfAttention(d_model=512, num_heads=8)
    out, attn = mha(x, causal=True)
    upper_tri_sum = torch.triu(attn, diagonal=1).sum().item()
    print(f"Output Shape  : {tuple(out.shape)} | Upper-Tri Causal Leak: {upper_tri_sum:.6f}")
    assert upper_tri_sum == 0.0
    print("[PASS] Part 1 Verified.")
    print("=" * 72)


if __name__ == "__main__":
    run_variance_preservation_test()