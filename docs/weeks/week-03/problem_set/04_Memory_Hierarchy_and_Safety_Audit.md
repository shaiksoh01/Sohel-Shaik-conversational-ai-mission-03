# Mission 03 — Part 4: Hardware Memory Hierarchy & Long-Context Safety Audit
**Deliverable:** `04_Memory_Hierarchy_and_Safety_Audit.md`  
**Starship Operational Unit:** USSE Noosphere Bridge Crew (Squadron Turing)  
**Galactic Waypoint:** Pleiades Open Cluster (`SEC-03-M45`, 444 ly from Sol)

---

## 1. FlashAttention IO Complexity Derivation

### 1.1 Standard Attention HBM Traffic: $\Theta(N d + N^2)$
1. **Score Materialization ($S = QK^T$):** Read $Q, K \in \mathbb{R}^{N \times d}$ ($2Nd$ reads) and write $S \in \mathbb{R}^{N \times N}$ to HBM ($N^2$ writes).
2. **Softmax ($P = \text{softmax}(S)$):** Read $S$ ($N^2$ reads) and write $P \in \mathbb{R}^{N \times N}$ to HBM ($N^2$ writes).
3. **Output Aggregation ($O = PV$):** Read $P$ and $V$ ($N^2 + Nd$ reads) and write $O \in \mathbb{R}^{N \times d}$ to HBM ($Nd$ writes).
- **Total Standard HBM Accesses:** $4Nd + 4N^2 = \Theta(Nd + N^2)$.

### 1.2 FlashAttention SRAM Block Tiling: $\Theta(Nd + \frac{N^2 d^2}{M})$
- Partition $K, V$ into $T_c = \lceil \frac{N}{B_c} \rceil = \Theta(\frac{Nd}{M})$ blocks where $B_c = \Theta(\frac{M}{d})$ fits in on-chip SRAM of size $M$.
- Outer loop loads $K_j, V_j$ once from HBM ($2Nd$ reads).
- Inner loop streams $Q_i, O_i$ ($\Theta(Nd)$ elements) across $T_c$ outer steps using online softmax rescaling without writing $S, P$ to HBM.
- **Total FlashAttention HBM Accesses:** $2Nd + T_c \cdot \Theta(Nd) = \Theta(Nd + \frac{N^2 d^2}{M})$.

---

## 2. KV-Cache Denial-of-Service (DoS) Simulation
- **Hardware Target:** 80 GB H100 GPU running 32-layer LLaMA-3 8B FP16 (16 GB weights + 4 GB overhead = **60 GB available for KV-Cache**).
- **Attack Vector 1 (MHA Flood):** 16 concurrent $N=32,768$ sessions under MHA require $16 \times 16\text{ GB} = 256\text{ GB}$, triggering CUDA OOM on the 4th session ($64\text{ GB} > 60\text{ GB}$).
- **Attack Vector 2 (Contiguous Slowloris under GQA):** Even with GQA ($4\text{ GB}$/session), static pre-allocation of $N_{\max}=32,768$ allows 16 idle sessions sending 16 tokens each to lock $16 \times 4\text{ GB} = 64\text{ GB}$ (>99.9% internal fragmentation), crashing AURA Subsystem 03.

---

## 3. Mitigation Strategy: PagedAttention & Multi-Head Latent Attention (MLA)
- **PagedAttention (vLLM):** Partitions KV-Cache into non-contiguous 16-token physical blocks allocated on demand, eliminating internal fragmentation and enabling copy-on-write prefix sharing.
- **Multi-Head Latent Attention (MLA / DeepSeek-V3):** Projects keys and values into a compressed low-rank latent vector ($d_c = 512$) plus decoupled RoPE key ($d_R = 64$), caching only $576$ elements per token/layer instead of $8192$ in MHA (**$14.2\times$ memory reduction**).