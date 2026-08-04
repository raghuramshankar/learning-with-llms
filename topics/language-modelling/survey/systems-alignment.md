# Literature Survey: Systems/Efficiency and Alignment for LLMs

Scope: kernels & memory, parallelism, inference serving, and alignment/post-training, for a
CS336-style "language modelling from scratch" explainer. Every paper below was checked against the
arXiv API (`http://export.arxiv.org/api/query`) with an exact `ti:"<title>"` query (script run
synchronously, `time.sleep(3-5)` between requests). Status is marked **VERIFIED** (exact title match
found via the API, id/date/authors confirmed) or **NOT FOUND** (query returned nothing under any
title variant tried — no id is invented in that case).

---

## 1. Kernels and memory

| Paper | First author (+et al.) | Year | arXiv ID | Status |
|---|---|---|---|---|
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao et al. | 2022 | **2205.14135** | VERIFIED |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Dao | 2023 | **2307.08691** | VERIFIED |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Shah et al. | 2024 | **2407.08608** | VERIFIED |
| Online normalizer calculation for softmax | Milakov & Gimelshein | 2018 | **1805.02867** | VERIFIED |
| Training Deep Nets with Sublinear Memory Cost | Chen et al. | 2016 | **1604.06174** | VERIFIED |

**Obstacle removed by each:**
- **FlashAttention** (Dao, Fu, Ermon, Rudra, Ré — 2205.14135, submitted 2022-05-27): standard
  attention materializes the full N×N attention/softmax matrix in slow HBM; FlashAttention fuses the
  QKᵀ→softmax→·V pipeline into a single kernel using tiling plus **online softmax** (recomputing the
  softmax normalizer incrementally, block by block, instead of needing a full second pass over each
  row) so the N×N matrix never touches HBM. **IO complexity**: standard attention needs
  Θ(N·d + N²) HBM accesses; FlashAttention needs **Θ(N²d²/M)** HBM accesses, where N = sequence
  length, d = head dimension, M = SRAM size (block size chosen ≈ M/(4d) so a Q/K/V/O tile fits on
  chip) — provably fewer accesses than standard attention whenever d ≪ M, which holds for realistic
  head dimensions. This reframes attention from "just reduce FLOPs" to an IO-aware algorithm design
  problem.
- **FlashAttention-2** (Dao — 2307.08691): restructures parallelization to also split across sequence
  length (not just batch/heads) and cuts non-matmul FLOPs, giving roughly 2× the throughput of
  FlashAttention-1 by better occupying GPU streaming multiprocessors and warps.
- **FlashAttention-3** (Shah et al. — 2407.08608): co-designed for Hopper (H100) — exploits
  **asynchrony** (warp-specialized producer/consumer pipelining that overlaps matmul with softmax
  using the Tensor Memory Accelerator) and **low precision** (FP8), reaching up to ~75% of H100's
  FP16 peak and ~1.2 PFLOPS/s in FP8, closing most of the remaining gap to hardware peak.
- **Online normalizer calculation for softmax** (Milakov & Gimelshein — 1805.02867): the direct
  precursor to FlashAttention's core trick — proves softmax's normalizing sum can be computed in one
  streaming pass with a running max and running (rescaled) sum, instead of the naive two-pass
  (max-then-sum) algorithm. This is exactly the "online softmax" recurrence FlashAttention reuses to
  fuse attention into a single kernel.
- **Training Deep Nets with Sublinear Memory Cost** (Chen, Xu, Zhang, Guestrin — 1604.06174): naive
  backprop caches every layer's activations, giving O(n) memory for an n-layer network. This paper
  shows caching only O(√n) checkpoints and recomputing the rest during the backward pass gives
  **O(√n) memory** at the cost of roughly one extra forward pass (~30% more compute) — the basis of
  gradient/activation checkpointing used throughout modern LLM training.

### Memory arithmetic for training (mixed-precision AdamW)

| Component | Bytes/param |
|---|---|
| bf16/fp16 parameters | 2 |
| bf16/fp16 gradients | 2 |
| fp32 master-copy parameters | 4 |
| fp32 Adam momentum (m) | 4 |
| fp32 Adam variance (v) | 4 |
| **Total** | **16 bytes/param** |

This is the "16Ψ bytes" figure the ZeRO paper uses directly (Ψ = parameter count): 4Ψ bytes for
fp16 params+grads, and 12Ψ bytes for the fp32 params+momentum+variance optimizer state.

### Roofline / arithmetic intensity

Arithmetic intensity (AI) = FLOPs / bytes moved from memory. A kernel is compute-bound if
AI > (peak FLOP/s)/(peak HBM bandwidth) — the **ridge point** — and memory-bandwidth-bound otherwise.
For an A100 (≈312 TFLOP/s bf16 tensor-core, ≈2 TB/s HBM), the ridge point is ≈156 FLOPs/byte.
Un-fused attention has low AI because it repeatedly reads/writes the N×N score matrix to HBM, making
it memory-bound despite being FLOP-heavy on paper — precisely the gap FlashAttention's fusion closes.

Decode-time autoregressive generation (batch small) has to re-read the *entire* parameter set (plus
KV cache) from HBM for every single new token produced, while doing only ~2×params FLOPs of work per
token — extremely low arithmetic intensity, so **decode is memory-bandwidth-bound**, not compute-bound;
throughput tracks HBM bandwidth / (bytes-per-param + KV-cache bytes), not peak FLOP/s. Prefill, in
contrast, processes the whole prompt in one pass with a large batch of tokens per weight read, so it is
comparatively compute-bound.

### GPU memory hierarchy (vendor spec sheets / general knowledge — not an arXiv claim, except where noted)

- **A100 (80GB SXM)**: ≈2.0 TB/s HBM2e bandwidth; 192 KB on-chip SRAM per SM (~20 MB aggregate across
  108 SMs); on-chip SRAM bandwidth ≈19 TB/s. These specific A100 numbers are the ones the
  FlashAttention paper (2205.14135) itself cites in its hardware background table.
- **H100 (80GB SXM5)**: ≈3.35 TB/s HBM3 bandwidth; 192 KB SRAM per SM.
- Ratio of SRAM : HBM bandwidth is roughly **9–10×** — the concrete reason IO-aware kernel design
  (tiling to keep working sets on-chip) matters so much for attention.

---

## 2. Parallelism

| Paper | First author (+et al.) | Year | arXiv ID | Status |
|---|---|---|---|---|
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari et al. | 2019 | **1910.02054** | VERIFIED |
| PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | Zhao et al. | 2023 | **2304.11277** | VERIFIED |
| Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | Shoeybi et al. | 2019 | **1909.08053** | VERIFIED |
| GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | Huang et al. | 2018 | **1811.06965** | VERIFIED |
| PipeDream: Fast and Efficient Pipeline Parallel DNN Training | Harlap/Narayanan et al. | 2018 | **1806.03377** | VERIFIED (found under this title, not "Generalized Pipeline Parallelism for DNN Training") |
| Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (introduces interleaved 1F1B) | Narayanan et al. | 2021 | **2104.04473** | VERIFIED |
| Zero Bubble Pipeline Parallelism | Qi et al. | 2023 | **2401.10241** | VERIFIED |
| Ring Attention with Blockwise Transformers for Near-Infinite Context | Liu et al. | 2023 | **2310.01889** | VERIFIED |

**Obstacle removed by each:**
- **ZeRO** (Rajbhandari, Rasley, Ruwase, He — 1910.02054): plain data parallelism replicates the
  entire optimizer state (16Ψ bytes, see above) on every GPU, capping model size at single-GPU memory.
  ZeRO **partitions** optimizer states / gradients / parameters across the data-parallel group instead
  of replicating them, re-materializing what's needed via collectives. With Ψ = params, N_d =
  data-parallel degree:
  - **Stage 1** (optimizer-state partitioning, P_os): **4Ψ + 12Ψ/N_d** bytes/GPU
  - **Stage 2** (+ gradient partitioning, P_os+g): **2Ψ + 14Ψ/N_d** bytes/GPU
  - **Stage 3** (+ parameter partitioning, P_os+g+p): **16Ψ/N_d** bytes/GPU — full linear (N_d-fold)
    reduction, at the cost of extra all-gather communication (~1.5× the communication volume of plain
    DP for stage 3, vs. the same volume as DP for stages 1–2).
- **PyTorch FSDP** (Zhao et al. — 2304.11277): productionizes ZeRO-3-style full sharding as a
  first-class, composable PyTorch primitive (rather than a separate library), documenting the
  engineering (e.g. prefetching, communication/computation overlap) needed to make it fast at scale.
- **Megatron-LM** (Shoeybi et al. — 1909.08053): a transformer layer's big weight matrices (MLP's
  h→4h and 4h→h projections, QKVO projections) can be **split across GPUs along columns/rows**
  (tensor/intra-layer model parallelism) with only two all-reduces per transformer block (one in
  attention, one in MLP) needed to resynchronize — lets model width exceed single-GPU memory without
  pipeline bubbles, at the cost of frequent, high-bandwidth communication (so it's normally confined
  to a fast intra-node NVLink domain).
- **GPipe** (Huang et al. — 1811.06965): partitions a model's *layers* across devices (pipeline
  parallelism) and splits each mini-batch into micro-batches so devices process different
  micro-batches concurrently, shrinking the idle "bubble" fraction to roughly O((K−1)/M) (K = pipeline
  stages, M = micro-batches per batch), trading recomputation (à la Chen et al. above) for memory.
- **PipeDream** (Harlap, Narayanan, Phanishayee et al. — 1806.03377): introduces **1F1B**
  (one-forward-one-backward) scheduling — interleaving forward and backward passes for different
  micro-batches instead of GPipe's "all-forward-then-all-backward" — so activation memory for a
  micro-batch is released as soon as its backward pass runs, instead of holding every in-flight
  micro-batch's activations simultaneously.
- **Narayanan et al. 2021** (2104.04473, the Megatron-LM 1F1B/interleaved-schedule + 3D-parallelism
  paper): generalizes 1F1B to an **interleaved** schedule (each device owns multiple non-contiguous
  pipeline stages) to further shrink the bubble, and lays out practical **3D parallelism**
  (data × tensor × pipeline) reasoning for large GPU clusters.
- **Zero Bubble Pipeline Parallelism** (Qi et al. — 2401.10241): observes the backward pass splits
  into two independent parts (gradient w.r.t. input vs. w.r.t. weights) and reorders them so the
  pipeline bubble can, in principle, be driven to **zero** without extra memory, beating 1F1B's
  residual bubble.
- **Ring Attention** (Liu, Zaharia, Abbeel — 2310.01889): shards a single very-long sequence's Q/K/V
  blocks across devices arranged in a ring, overlapping the ring communication of K/V blocks with
  (FlashAttention-style) blockwise attention compute — sequence/context parallelism that lets
  effective context length scale with device count, bounded by aggregate cluster memory rather than
  any single device's memory, with communication hidden behind compute.

### Communication-cost reasoning, in brief
- **Data parallelism**: all-reduce gradients every step, cost ∝ model size, independent of batch size.
- **Tensor parallelism**: 2 all-reduces per transformer block, cost ∝ activation size × depth — kept
  inside a fast NVLink domain.
- **Pipeline parallelism**: point-to-point send/recv of activations/gradients between adjacent stages
  only, cost ∝ micro-batch activation size, cheap enough to cross slower inter-node links, at the cost
  of bubble overhead.
- **Sequence/context parallelism (Ring Attention)**: ring point-to-point of K/V blocks, overlapped
  with compute.
- **3D/4D parallelism**: combines data × tensor × pipeline (× sequence/expert) parallelism, matching
  each dimension to the fastest link it needs (tensor→intra-node NVLink, pipeline/data→inter-node).

---

## 3. Inference

| Paper | First author (+et al.) | Year | arXiv ID | Status |
|---|---|---|---|---|
| Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon et al. | 2023 | **2309.06180** | VERIFIED |
| Orca: A Distributed Serving System for Transformer-Based Generative Models | Yu et al. | 2022 | — | **NOT FOUND** (OSDI 2022 paper; no arXiv preprint located under this or related titles) |
| Fast Inference from Transformers via Speculative Decoding | Leviathan et al. | 2022 | **2211.17192** | VERIFIED |
| Accelerating Large Language Model Decoding with Speculative Sampling | Chen et al. | 2023 | **2302.01318** | VERIFIED |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. | 2022 | **2210.17323** | VERIFIED |
| AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | Lin et al. | 2023 | **2306.00978** | VERIFIED |
| SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | Xiao et al. | 2022 | **2211.10438** | VERIFIED |
| LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | Dettmers et al. | 2022 | **2208.07339** | VERIFIED |
| s1: Simple test-time scaling | Muennighoff et al. | 2025 | **2501.19393** | VERIFIED |

**Prefill vs. decode.** Prefill processes the whole prompt in one parallel pass — large token batch per
weight read — so it's compute-bound. Decode generates one token at a time, autoregressively; each step
must re-read the whole parameter set and the whole KV cache from HBM to produce a single new token, so
arithmetic intensity collapses and **decode is memory-bandwidth-bound**: throughput ≈ HBM bandwidth /
(bytes-per-param-read + per-token KV-cache bytes), essentially independent of the GPU's peak FLOP/s.

**KV cache size formula:**
```
KV_bytes = 2 * n_layers * n_kv_heads * d_head * seq_len * batch * bytes_per_element
```
(the factor 2 is for K and V; with multi-head attention and no GQA/MQA, n_kv_heads*d_head = hidden_size,
so this is equivalently `2 * n_layers * hidden_size * seq_len * batch * bytes_per_element`). This is
the exact quantity **PagedAttention/vLLM** (Kwon et al., 2309.06180) attacks: naive contiguous KV-cache
allocation wastes 60–80% of allocated memory to fragmentation/over-reservation for the (unknown at
alloc time) generation length; PagedAttention manages the KV cache in fixed-size, non-contiguous
"pages" (borrowing virtual-memory paging from OS design), letting vLLM batch far more concurrent
sequences in the same HBM budget.

**Obstacle removed by each of the rest:**
- **Orca** (Yu et al., OSDI 2022 — not found on arXiv): the obstacle it removed was
  **request-level (static) batching**, where a batch can't start new requests or return finished ones
  until every sequence in the batch is done; Orca introduced **iteration-level (continuous) scheduling**
  — a new request can join, and a finished one can leave, at every decoding step — dramatically
  improving GPU utilization and throughput for serving. (This idea is now standard in vLLM/TGI/etc.,
  even though the original Orca paper itself is not on arXiv.)
- **Speculative decoding** (Leviathan, Kalman, Matias — 2211.17192) and **speculative sampling**
  (Chen, Borgeaud, Irving et al., DeepMind — 2302.01318): decoding one token at a time is
  memory-bandwidth-bound and cannot use spare compute. Both papers independently propose using a small,
  cheap "draft" model to propose several tokens speculatively, then verifying all of them in a single
  parallel forward pass of the large target model, accepting/rejecting via a rejection-sampling scheme
  that provably preserves the target model's exact output distribution — trading idle compute for fewer
  memory-bound decode steps, giving 2–3× wall-clock speedups with no quality loss.
- **GPTQ** (Frantar, Ashkboos, Hoefler, Alistarh — 2210.17323): removed the need for retraining/QAT to
  compress a pretrained LLM — a one-shot, layer-by-layer weight quantization using approximate
  second-order (Hessian) information, quantizing to 3–4 bits with minimal accuracy loss on
  multi-billion-parameter models in a few GPU-hours.
- **AWQ** (Lin, Tang, Tang et al. — 2306.00978): observes that a small fraction of "salient" weight
  channels (identified by activation magnitude, not weight magnitude) dominate quantization error;
  protects them via per-channel scaling before quantizing, without needing backprop or a calibration
  reconstruction step like GPTQ — faster to run and more robust across domains.
- **SmoothQuant** (Xiao, Lin, Seznec et al. — 2211.10438): activations (not weights) have large,
  hard-to-quantize outlier channels in LLMs, blocking accurate INT8 activation quantization.
  SmoothQuant migrates quantization difficulty from activations to weights via a per-channel scaling
  factor, enabling accurate **W8A8** (8-bit weights and activations) inference.
- **LLM.int8()** (Dettmers, Lewis, Belkada, Zettlemoyer — 2208.07339): shows outlier feature
  dimensions in large models break naive INT8 quantization; introduces mixed-precision decomposition
  (INT8 for the bulk of the matrix, FP16 for outlier dimensions) enabling INT8 inference on models up
  to 175B parameters with no performance degradation.
- **s1 / budget forcing** (Muennighoff et al. — 2501.19393): shows a small (32B) model, fine-tuned on
  just 1,000 curated reasoning traces, can be pushed to match/exceed o1-style performance purely by
  controlling **test-time compute** via "budget forcing" — forcibly appending "Wait" to extend the
  model's chain-of-thought when it tries to stop early, or forcibly inserting an end-of-thinking token
  to cut reasoning short — showing inference-time compute scaling is a controllable, largely
  training-free lever distinct from RL-based reasoning training (contrast with DeepSeek-R1 below,
  which gets extended reasoning via RL).

---

## 4. Alignment

| Paper | First author (+et al.) | Year | arXiv ID | Status |
|---|---|---|---|---|
| Finetuned Language Models Are Zero-Shot Learners (FLAN) | Wei et al. | 2021 | **2109.01652** | VERIFIED |
| Multitask Prompted Training Enables Zero-Shot Task Generalization (T0) | Sanh et al. | 2021 | **2110.08207** | VERIFIED |
| Self-Instruct: Aligning Language Models with Self-Generated Instructions | Wang et al. | 2022 | **2212.10560** | VERIFIED |
| Alpaca: A Strong, Replicable Instruction-Following Model | Taori et al. | 2023 | — | **NOT FOUND** (Stanford blog post + GitHub release; no formal arXiv paper exists) |
| Learning to summarize from human feedback | Stiennon et al. | 2020 | **2009.01325** | VERIFIED |
| Proximal Policy Optimization Algorithms (PPO) | Schulman et al. | 2017 | **1707.06347** | VERIFIED |
| Training language models to follow instructions with human feedback (InstructGPT) | Ouyang et al. | 2022 | **2203.02155** | VERIFIED |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO) | Rafailov et al. | 2023 | **2305.18290** | VERIFIED |
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (introduces GRPO) | Shao et al. | 2024 | **2402.03300** | VERIFIED |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI et al. | 2025 | **2501.12948** | VERIFIED |
| Constitutional AI: Harmlessness from AI Feedback | Bai et al. | 2022 | **2212.08073** | VERIFIED |
| Tulu 3: Pushing Frontiers in Open Language Model Post-Training | Lambert et al. | 2024 | **2411.15124** | VERIFIED |

**Obstacle removed by each:**
- **FLAN** (Wei et al. — 2109.01652): shows that instruction-tuning a pretrained LM on a mixture of
  tasks phrased as natural-language instructions substantially improves **zero-shot** generalization
  to unseen tasks — the base capability that later RLHF pipelines are built on top of.
- **T0** (Sanh et al. — 2110.08207): concurrently shows the same multitask-prompted zero-shot
  generalization result with an explicit focus on held-out task generalization and prompt diversity.
- **Self-Instruct** (Wang et al. — 2212.10560): removes the bottleneck of needing large amounts of
  human-written instruction data — bootstraps an instruction-tuning dataset by having a model generate
  its own instructions, inputs, and outputs from a small seed set, then filtering for quality/diversity.
- **Alpaca** (Taori et al., not on arXiv): popularized applying Self-Instruct-style data generation
  (using OpenAI's `text-davinci-003`) to cheaply instruction-tune a small open model (LLaMA-7B),
  demonstrating instruction-following behavior could be replicated at low cost — but as a blog
  post/GitHub release rather than a peer-reviewed/arXiv paper.
- **Learning to summarize from human feedback** (Stiennon, Ouyang, Wu et al. — 2009.01325): the direct
  precursor to InstructGPT — establishes the reward-model + PPO RLHF recipe (train a reward model from
  human pairwise comparisons, then optimize a policy against it with PPO) on the summarization task,
  showing it beats supervised fine-tuning alone on human preference.
- **PPO** (Schulman et al. — 1707.06347): the clipped surrogate-objective policy-gradient algorithm
  that both this paper's summarization work and InstructGPT use as the RL optimizer — provides a
  simple, stable, first-order alternative to trust-region methods (TRPO) for policy optimization.
- **InstructGPT** (Ouyang et al. — 2203.02155): scales the RLHF recipe (SFT → reward model → PPO) to
  general instruction-following, not just summarization, and formalizes the exact PPO-with-KL-penalty
  RL objective the field now calls "RLHF" (full objective given in the formulas section below) —
  removes the mismatch between what pretraining/SFT optimizes (next-token likelihood on a fixed corpus)
  and what users actually want (helpful, honest, harmless responses to arbitrary instructions).
- **DPO** (Rafailov, Sharma, Mitchell et al. — 2305.18290): removes the need for a separate reward
  model and an online RL loop entirely — derives a closed-form loss (below) directly over preference
  pairs by exploiting the closed-form relationship between the optimal RLHF policy and its implicit
  reward, cutting the alignment pipeline's complexity and training instability at the cost of some
  flexibility (offline preference data only, no online exploration).
- **DeepSeekMath / GRPO** (Shao et al. — 2402.03300): removes the need for a **separate value/critic
  network** (roughly doubling memory/compute of PPO) — estimates advantage by normalizing a group of
  sampled rewards for the same prompt against each other (formula below) instead of a learned
  value function, making RL post-training substantially cheaper.
- **DeepSeek-R1** (DeepSeek-AI et al. — 2501.12948): shows large-scale **RLVR** (reinforcement
  learning with verifiable rewards — e.g. checking a math answer or code test pass/fail, with no
  learned reward model at all) applied with GRPO to a base model can elicit long chain-of-thought,
  self-verification, and reflection behavior ("aha moments") purely from the RL signal, without any
  supervised reasoning-trace data as a cold start (in the R1-Zero variant) — removes the assumption
  that step-by-step reasoning ability must be taught via SFT on human/distilled reasoning traces.
- **Constitutional AI** (Bai et al. — 2212.08073): removes much of the need for human-labeled harmful
  vs. harmless comparison data — a model critiques and revises its own outputs against a written set of
  principles (a "constitution"), and this AI-generated preference data (RLAIF) is then used the same
  way human preference data is used in RLHF.
- **Tulu 3** (Lambert et al. — 2411.15124): an open, fully-documented modern post-training recipe
  (SFT + DPO + a novel RLVR stage) explicitly built to close the gap between open and closed-model
  post-training pipelines, with public data/code/recipes — representative of current best-practice
  "combine everything" post-training (see Section 5).

---

## 5. Current SOTA and contested questions (2025 – mid-2026)

| Paper | First author (+et al.) | Date | arXiv ID | Status |
|---|---|---|---|---|
| Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | Yue et al. | Apr 2025 | **2504.13837** | VERIFIED |
| SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training | Chu et al. | Jan 2025 | **2501.17161** | VERIFIED |
| ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models | Liu et al. | May 2025 | **2505.24864** | VERIFIED |
| Spurious Rewards: Rethinking Training Signals in RLVR | Shao et al. | Jun 2025 | **2506.10947** | VERIFIED |
| RL's Razor: Why Online Reinforcement Learning Forgets Less | Shenfeld et al. | Sep 2025 | **2509.04259** | VERIFIED |
| Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward | Chen et al. | Dec 2025 | **2512.16912** | VERIFIED |
| Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs | Yan et al. | Jan 2026 | **2601.11061** | VERIFIED |

**The contested question — does RL add capability, or just sharpen sampling?**
- **Yue et al. (2504.13837)**, the paper that opened this debate, measures pass@k at large k for
  RLVR-trained reasoning models vs. their base models on math/code benchmarks, and finds the *base*
  model's pass@k eventually catches up to (or exceeds) the RL model's — i.e. the reasoning paths RL
  models produce were already reachable by the base model's sampling distribution. Their reading: RLVR
  mainly **improves sampling efficiency** (biases the model toward correct paths it could already
  generate) rather than expanding the base model's underlying reasoning *capacity*/coverage.
- **SFT Memorizes, RL Generalizes (2501.17161)** independently supports part of this picture from the
  opposite angle — showing RL-trained policies generalize better to novel task variants than SFT,
  while SFT tends to memorize training-distribution patterns — i.e. RL's benefit is about
  *robustness/generalization* of what's already learned, consistent with a "sharpening" rather than
  "new capability" story, though framed as a positive result for RL.
- **ProRL (2505.24864)** is a direct rebuttal in spirit: it argues the "RL doesn't expand capacity"
  finding is an artifact of *insufficient training* — with prolonged RL training (thousands of steps,
  not the few hundred typical in earlier studies), plus techniques to maintain exploration/entropy,
  models solve problems the base model *cannot* solve at any sampled k, i.e. genuinely new reasoning
  boundaries are reached, not just resampled.
- **Spurious Rewards (2506.10947)** complicates the picture further from yet another angle: it shows
  RLVR on Qwen2.5-Math models improves performance even with **random or spurious reward signals**
  (rewarding outputs that merely contain certain formatting, unrelated to correctness), suggesting for
  some models/setups the RL stage is mostly eliciting/amplifying reasoning behaviors already latent
  from pretraining, largely independent of whether the reward signal is even meaningful — a strong
  caution against attributing RLVR gains purely to "the reward taught the model to reason."
- **RL's Razor (2509.04259)** offers a mechanistic explanation for why RL-tuned models forget less of
  their pretrained capability than SFT-tuned models: RL implicitly stays closer (in a KL sense) to the
  base policy because it only reinforces trajectories the model already assigns nonzero probability to,
  whereas SFT can push weights toward off-distribution targets — reframing "RL vs SFT" as a
  KL-locality difference rather than a capability-vs-no-capability difference.
- **Exploration vs Exploitation (2512.16912)** and **Spurious Rewards Paradox (2601.11061)** (Dec
  2025 / Jan 2026, i.e. the most current entries in this survey) continue the thread mechanistically —
  respectively re-examining clipping/entropy dynamics in RLVR's optimization, and showing spurious
  rewards can specifically activate **memorization shortcuts** rather than genuine reasoning
  improvements, i.e. some of the field's apparent "RLVR works even with noise" results reflect the
  model exploiting spurious cues rather than reasoning gains — evidence the debate is still very much
  open at the time of writing.

**Current best practice, as reflected in these sources:**
- **Serving**: continuous/iteration-level batching (Orca-style scheduling) + paged, non-contiguous KV
  cache management (PagedAttention/vLLM) is the baseline for any serving stack; quantization
  (GPTQ/AWQ/SmoothQuant-family, or FP8/INT4 native) and speculative decoding are layered on top to cut
  the memory-bandwidth cost of decode specifically.
- **Post-training**: the Tulu 3 / DeepSeek-R1 pattern — SFT (often on synthetic/distilled reasoning
  traces) → preference optimization (DPO or PPO-based RLHF for general instruction-following/safety)
  → RLVR with GRPO-style group-relative advantages on verifiable domains (math, code) — is the closest
  thing to current consensus best practice, though exactly *why* the RLVR stage helps (new capability
  vs. sharpened sampling vs. partly-spurious signal) remains an open, actively contested research
  question as of early 2026.

---

## 6. Key numbers and formulas (for quiz questions)

1. **FlashAttention IO complexity**: standard attention needs **Θ(N·d + N²)** HBM accesses; FlashAttention
   needs **Θ(N²d²/M)** HBM accesses (N = sequence length, d = head dim, M = SRAM size, block size ≈
   M/(4d)). [Dao et al., 2205.14135]

2. **Online softmax recurrence** (Milakov & Gimelshein, 1805.02867 — the trick FlashAttention fuses
   in): maintaining running max `m_i` and running (rescaled) sum `l_i` over blocks of logits lets the
   full softmax normalizer be computed in a single pass:
   `m_new = max(m_old, max(x_block))`, `l_new = l_old * exp(m_old - m_new) + sum(exp(x_block - m_new))`.

3. **Bytes per parameter, mixed-precision AdamW training**: **16 bytes/param** total —
   2 (fp16/bf16 param) + 2 (fp16/bf16 grad) + 4 (fp32 master param) + 4 (fp32 Adam m) + 4 (fp32 Adam v).
   [ZeRO paper convention, Rajbhandari et al., 1910.02054]

4. **ZeRO per-stage memory** (Ψ = params, N_d = data-parallel degree), vs. 16Ψ baseline (full replication):
   Stage 1 (P_os): **4Ψ + 12Ψ/N_d**; Stage 2 (P_os+g): **2Ψ + 14Ψ/N_d**; Stage 3 (P_os+g+p): **16Ψ/N_d**
   (up to N_d-fold reduction). [1910.02054]

5. **KV cache size**: `KV_bytes = 2 * n_layers * n_kv_heads * d_head * seq_len * batch * bytes_per_element`
   (factor 2 for K and V). This is the quantity PagedAttention (2309.06180) manages via fixed-size pages
   instead of contiguous over-allocation.

6. **A100 / H100 hardware numbers** (vendor specs, cited in FlashAttention's hardware table for A100):
   A100 80GB SXM — **~2.0 TB/s HBM2e bandwidth**, **192 KB SRAM/SM** (~19 TB/s on-chip bandwidth).
   H100 80GB SXM5 — **~3.35 TB/s HBM3 bandwidth**, **192 KB SRAM/SM**. SRAM:HBM bandwidth ratio ≈ 9-10×.

7. **InstructGPT PPO-ptx objective** (Ouyang et al., 2203.02155, Sec. 3.5 Eq. 2):
   ```
   objective(phi) = E_(x,y)~D_piRL_phi [ r_theta(x,y) - beta * log( piRL_phi(y|x) / piSFT(y|x) ) ]
                     + gamma * E_x~D_pretrain [ log( piRL_phi(x) ) ]
   ```
   First term: reward-model score minus a per-token **KL penalty** (coefficient β) against the SFT
   policy, optimized via PPO — bounds reward-model over-optimization/reward hacking. Second term (the
   "ptx" term): mixes in pretraining log-likelihood, weighted by γ, to control the "alignment tax."

8. **DPO closed-form loss** (Rafailov et al., 2305.18290, Eq. 7):
   ```
   L_DPO(pi_theta; pi_ref) = - E_(x,y_w,y_l)~D [ log sigmoid( beta * log(pi_theta(y_w|x)/pi_ref(y_w|x))
                                                              - beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)) ) ]
   ```
   Derivation sketch: the RLHF objective `max_pi E[r(x,y)] - beta*KL(pi||pi_ref)` has closed-form
   optimum `pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y)/beta)`; inverting gives
   `r(x,y) = beta*log(pi*(y|x)/pi_ref(y|x)) + beta*log Z(x)`; substituting into the Bradley-Terry
   preference model `p(y_w > y_l) = sigmoid(r(y_w) - r(y_l))` cancels the intractable partition
   function Z(x), leaving a loss purely over policy log-ratios — no explicit reward model, no RL loop.

9. **GRPO group-relative advantage** (Shao et al./DeepSeekMath, 2402.03300): for a group of G outputs
   {o_1,...,o_G} sampled per prompt with rewards {r_1,...,r_G}:
   ```
   A_i = ( r_i - mean(r_1,...,r_G) ) / std(r_1,...,r_G)
   ```
   used in a PPO-style clipped objective with an explicit KL penalty term against a reference policy
   (no learned value/critic network needed, unlike PPO) — this is the RL algorithm behind DeepSeek-R1
   (2501.12948).

---

## Sources not found on arXiv (flagged, not fabricated)

- **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et al., OSDI 2022)
  — no arXiv preprint located despite several query variants; it is a systems (OSDI) paper without an
  arXiv release.
- **Alpaca: A Strong, Replicable Instruction-Following Model** (Taori et al., 2023) — released as a
  Stanford CRFM blog post + GitHub repo, never published as a formal arXiv paper.
- **PipeDream** — the exact title "PipeDream: Generalized Pipeline Parallelism for DNN Training"
  returned no match; the paper *is* on arXiv under the title "PipeDream: Fast and Efficient Pipeline
  Parallel DNN Training" (1806.03377), which is the one cited above.
