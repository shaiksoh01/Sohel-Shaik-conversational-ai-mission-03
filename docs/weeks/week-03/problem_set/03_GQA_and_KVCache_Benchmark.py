"""
Mission 03 - Part 3: Grouped-Query Attention (GQA) & KV-Cache Benchmarking
Deliverable: 03_GQA_and_KVCache_Benchmark.py
Starship Operational Unit: USSE Noosphere Bridge Crew (Squadron Turing)
"""

import math
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    def __init__(self):
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache is None:
            self.k_cache, self.v_cache = k_new, v_new
        else:
            self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)
        return self.k_cache, self.v_cache


class GQAAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_q_heads: int = 32, num_kv_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_Q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(num_q_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, use_causal_mask: bool = True):
        b, seq_len, _ = x.shape
        q = self.W_Q(x).view(b, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        total_len = k.size(2)
        if self.num_queries_per_kv > 1:
            k = torch.repeat_interleave(k, repeats=self.num_queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, repeats=self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if use_causal_mask and seq_len > 1:
            mask = torch.triu(
                torch.ones(seq_len, total_len, device=x.device, dtype=torch.bool),
                diagonal=total_len - seq_len + 1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.W_O(ctx)


def run_benchmark():
    print("=" * 78)
    print("PART 3: GQA & KV-CACHE BENCHMARK (32 LAYERS, FP16, d_k=128)")
    print("=" * 78)
    print(f"{'Architecture':<14} | {'SeqLen (N)':>10} | {'KV-Cache (MB)':>14} | {'BW @ 50t/s (GB/s)':>18} | {'Decode (ms)':>12}")
    print("-" * 78)

    torch.manual_seed(42)
    for arch_name, h_kv in [("MHA (32:32)", 32), ("GQA (32:8)", 8), ("MQA (32:1)", 1)]:
        layer = GQAAttention(d_model=512, num_q_heads=32, num_kv_heads=h_kv).eval()
        q_tok = torch.randn(1, 1, 512)
        for n in [1024, 4096, 16384, 32768]:
            kv_bytes = 2 * 32 * n * h_kv * 128 * 2
            kv_mb = kv_bytes / (1024 ** 2)
            bw_gb_s = (kv_bytes * 50.0) / (1024 ** 3)

            cache = KVCache()
            with torch.no_grad():
                cache.k_cache = torch.randn(1, h_kv, n - 1, 16)
                cache.v_cache = torch.randn(1, h_kv, n - 1, 16)
                t0 = time.perf_counter()
                _ = layer(q_tok, kv_cache=cache, use_causal_mask=False)
                ms = (time.perf_counter() - t0) * 1000.0
            print(f"{arch_name:<14} | {n:>10d} | {kv_mb:>14.2f} | {bw_gb_s:>18.2f} | {ms:>12.3f}")
        print("-" * 78)


if __name__ == "__main__":
    run_benchmark()