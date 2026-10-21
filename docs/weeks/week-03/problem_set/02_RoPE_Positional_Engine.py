"""
Mission 03 - Part 2: Rotary Positional Embeddings (RoPE) Engine
Deliverable: 02_RoPE_Positional_Engine.py
Starship Operational Unit: USSE Noosphere Bridge Crew (Squadron Turing)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def precompute_freqs_cis(dim: int, end: int, base: float = 10000.0):
    assert dim % 2 == 0, "RoPE dimension must be even"
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, freqs_cis) -> torch.Tensor:
    cos, sin = freqs_cis
    batch, num_heads, seq_len, head_dim = x.shape

    x_pairs = x.float().reshape(batch, num_heads, seq_len, head_dim // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    cos_b = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin_b = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    rot_x1 = x1 * cos_b - x2 * sin_b
    rot_x2 = x1 * sin_b + x2 * cos_b

    x_rotated = torch.stack([rot_x1, rot_x2], dim=-1).reshape(batch, num_heads, seq_len, head_dim)
    return x_rotated.type_as(x)


def verify_and_plot_rope():
    torch.manual_seed(42)
    head_dim, max_pos = 128, 2048
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=max_pos)

    q_seq = torch.randn(1, 1, 1, head_dim).expand(1, 1, max_pos, head_dim).clone()
    k_seq = torch.randn(1, 1, 1, head_dim).expand(1, 1, max_pos, head_dim).clone()
    q_rot = apply_rope(q_seq, freqs_cis)
    k_rot = apply_rope(k_seq, freqs_cis)

    print("=" * 74)
    print("PART 2: RoPE RELATIVE DISTANCE INVARIANCE VALIDATION")
    print("=" * 74)
    for m, n, delta in [(12, 5, 100), (50, 180, 420), (0, 256, 1024), (333, 333, 500)]:
        d1 = torch.sum(q_rot[0, 0, m] * k_rot[0, 0, n]).item()
        d2 = torch.sum(q_rot[0, 0, m + delta] * k_rot[0, 0, n + delta]).item()
        err = abs(d1 - d2)
        assert err < 1e-4, f"Invariance error too large: {err}"
        print(
            f"m={m:<3d} n={n:<3d} delta={delta:<4d} | "
            f"<q_m, k_n>={d1:9.4f} | <q_m+d, k_n+d>={d2:9.4f} | err={err:.2e}"
        )

    max_dist = 512
    freqs_sub = (freqs_cis[0][:max_dist], freqs_cis[1][:max_dist])
    ones_rot = apply_rope(torch.ones(1, 1, max_dist, head_dim), freqs_sub)
    rope_kernel = torch.matmul(ones_rot[0, 0], ones_rot[0, 0, 0]) / head_dim
    distances = torch.arange(max_dist)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(distances.tolist(), rope_kernel.tolist(), label="RoPE Trigonometric Decay", color="#1f77b4")
    axes[0].plot(distances.tolist(), torch.exp(-(1 / 16) * distances.float()).tolist(), "--", label="ALiBi (m=1/16)", color="#ff7f0e")
    axes[0].plot(distances.tolist(), torch.exp(-(1 / 2) * distances.float()).tolist(), ":", label="ALiBi (m=1/2)", color="#d62728")
    axes[0].set_title("RoPE Relative Decay vs. ALiBi Linear Penalty")
    axes[0].set_xlabel("Relative Token Distance |m - n|")
    axes[0].set_ylabel("Normalized Coupling Weight")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    n_grid = 128
    sub_rot = ones_rot[0, 0, :n_grid]
    toeplitz = (torch.matmul(sub_rot, sub_rot.T) / head_dim).tolist()
    im = axes[1].imshow(toeplitz, cmap="viridis", origin="upper")
    axes[1].set_title("RoPE 2D Pairwise Attention Coupling Heatmap")
    axes[1].set_xlabel("Key Position (n)")
    axes[1].set_ylabel("Query Position (m)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("rope_vs_alibi_decay.png", dpi=150)
    plt.close(fig)
    print("[PASS] Part 2 Verified & 'rope_vs_alibi_decay.png' saved.")
    print("=" * 74)


if __name__ == "__main__":
    verify_and_plot_rope()