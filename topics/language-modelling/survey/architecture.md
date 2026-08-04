# Literature survey: Transformer Architecture

Surveyed 2026-08-04 via the arXiv API (`http://export.arxiv.org/api/query`, `ti:"<exact title>"`
title-verification queries, rate-limited ~1 req/8s with retry/backoff), plus two primary-source
PDF/HTML fetches (Vaswani et al. 2017, Shazeer 2020) to pull verbatim quotes and exact numbers, and
one web search cross-check for Llama 3's per-model hyperparameter table. Raw material for the
learning-new-topic explainer build.

**Verification key:** ✅ VERIFIED = confirmed via an arXiv API `ti:` exact-title query (arXiv id,
authors, submission date all pulled from the API response). ❌ NOT FOUND = queried and arXiv
returned zero results; no id is given because none exists to give. Every arXiv id in this document
came from an actual API response — none were recalled from memory and typed in.

---

## 1. Foundations

### 1.1 Attention Is All You Need — THE seminal paper

✅ **Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017), *Attention Is All
You Need*, arXiv:1706.03762** (submitted 2017-06-12, v7 2023-08-02). Obstacle removed: replaced
recurrence and convolution entirely with a stacked self-attention architecture, removing the
sequential-computation bottleneck that made RNN/LSTM encoder-decoders slow to train and hard to
parallelize.

**Scaled dot-product attention** (Section 3.2.1), exact formula as published:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

**Their stated reason for the 1/√d_k factor** (verbatim quote, pulled directly from the paper text):

> "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the
> softmax function into regions where it has extremely small gradients."

with the accompanying footnote making the variance argument explicit (also verbatim):

> "To illustrate why the dot products get large, assume that the components of q and k are
> independent random variables with mean 0 and variance 1. Then their dot product,
> $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$, has mean 0 and variance $d_k$."

So without the scale factor, attention logits have variance $d_k$ (growing with head dimension);
dividing by $\sqrt{d_k}$ renormalizes the variance back to ~1, keeping the softmax out of its
saturated, near-zero-gradient regime.

Base-model config (also verified from the paper): $d_{model}=512$, $h=8$ heads, $d_k=d_v=64$
($d_k = d_v = d_{model}/h$).

### 1.2 Precursors — why attention was invented before the Transformer existed

✅ **Bahdanau, Cho, Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and
Translate*, arXiv:1409.0473** (2014-09-01). Obstacle removed: vanilla encoder-decoder RNNs compress
the entire source sentence into one fixed-length vector, a bottleneck that degrades badly on long
sentences; this paper introduced a learned soft-alignment ("attention") over all encoder hidden
states at each decoding step, letting the decoder look back at relevant source words directly. This
is the direct conceptual ancestor of Transformer attention.

✅ **Sutskever, Vinyals, Le (2014), *Sequence to Sequence Learning with Neural Networks*,
arXiv:1409.3215** (2014-09-10). Obstacle removed: showed a general-purpose, end-to-end deep network
(a multilayered LSTM encoder-decoder) could map variable-length input sequences to variable-length
output sequences with minimal task-specific engineering — established the encoder-decoder paradigm
that Bahdanau attention, and later the Transformer, improved on.

### 1.3 GPT-1 / GPT-2 / GPT-3

❌ **GPT-1 — Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* —
NOT FOUND on arXiv.** Queried `ti:"Improving Language Understanding by Generative Pre-Training"`;
zero results. This is expected: OpenAI published GPT-1 only as a PDF technical report on their own
website, never submitted to arXiv. (Informally, its contribution was showing that generative
pre-training of a decoder-only Transformer, followed by task-specific discriminative fine-tuning,
transfers well across diverse NLP tasks — but this claim is *not* arXiv-sourced.)

❌ **GPT-2 — Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* — NOT
FOUND on arXiv.** Queried `ti:"Language Models are Unsupervised Multitask Learners"`; zero results.
Same situation: OpenAI-hosted PDF/blog only, never an arXiv submission. (Informally: demonstrated
zero-shot task transfer purely from scaling a decoder-only LM's parameters/data, no task-specific
fine-tuning needed — again, not arXiv-sourced, flagged accordingly.)

✅ **Brown et al. (2020), *Language Models are Few-Shot Learners*, arXiv:2005.14165** (2020-05-28,
GPT-3). Obstacle removed: showed that in-context few-shot learning — conditioning on a handful of
examples in the prompt, with zero gradient updates — improves smoothly with model scale, removing
the need for task-specific fine-tuning datasets for a large class of tasks.

### 1.4 BERT — for the encoder vs. decoder-only contrast

✅ **Devlin, Chang, Lee, Toutanova (2018), *BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding*, arXiv:1810.04805** (2018-10-11). Obstacle removed: GPT-style decoder-only
LMs are trained left-to-right (causal), so each token can only see prior context; BERT uses an
**encoder-only** stack with a masked-language-model objective (predict randomly masked tokens from
both left and right context), enabling deep *bidirectional* representations — at the cost of not
being a natural generator. This is the canonical encoder-only vs. decoder-only architectural fork:
BERT-style encoders excel at understanding/classification tasks (fine-tune + one output layer), while
GPT-style decoders excel at open-ended generation and, later, in-context few-shot learning.

---

## 2. Modern architecture components

### 2.1 Normalization

✅ **Ba, Kiros, Hinton (2016), *Layer Normalization*, arXiv:1607.06450** (2016-07-21). Obstacle
removed: BatchNorm's statistics depend on the mini-batch and are awkward for RNNs/variable-length
sequences; LayerNorm normalizes across the *feature* dimension per example instead, removing the
batch-size dependency.

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot g + b, \quad \mu = \tfrac{1}{d}\sum_i x_i,\ \ \sigma^2 = \tfrac{1}{d}\sum_i (x_i-\mu)^2$$

✅ **Zhang & Sennrich (2019), *Root Mean Square Layer Normalization*, arXiv:1910.07467**
(2019-10-16). Obstacle removed: hypothesized (and confirmed) that LayerNorm's *re-centering*
(mean-subtraction) is dispensable — only the *re-scaling* matters for the stabilizing effect —
so RMSNorm drops the mean/bias terms entirely, cutting compute and latency at equal quality. Now
the default normalization in LLaMA, Mistral, DeepSeek, Qwen, and effectively every 2023+ open LLM.

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot g, \quad \text{RMS}(x) = \sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}$$

✅ **Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu (2020), *On Layer Normalization in the
Transformer Architecture*, arXiv:2002.04745** (2020-02-12). Obstacle removed: proved via mean-field
theory that in the *original* (Post-LN) Transformer, gradients at initialization are large near the
output layer, forcing a fragile, hyperparameter-sensitive learning-rate warm-up phase just to avoid
divergence. Moving the normalization *inside* the residual branch (Pre-LN) keeps gradient norms
well-behaved from initialization, so warm-up becomes unnecessary and training is far more stable at
scale. Virtually every modern LLM (GPT-2 onward, LLaMA, everything below) uses Pre-LN.

$$\text{Post-LN: } x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l)) \qquad \text{Pre-LN: } x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))$$

✅ **Henry, Dachapally, Pawar, Chen (2020), *Query-Key Normalization for Transformers*,
arXiv:2010.04245** (2020-10-08). Obstacle removed: unbounded query/key dot products can push softmax
into saturation (the same failure mode Section 1.1's scaling factor targets, but arising *during
training* as weight norms grow); QKNorm applies $\ell_2$-normalization along the head dimension to Q
and K before the dot product, then rescales by a single learned per-head scalar, keeping logits
bounded regardless of weight-norm growth.

✅ **Dehghani, Djolonga, Mustafa et al. (2023), *Scaling Vision Transformers to 22 Billion
Parameters*, arXiv:2302.05442** (2023-02-10). Obstacle removed: at extreme parameter scale (22B),
plain attention becomes unstable — logits diverge and training loss spikes; the paper found QK-Norm
essential for stability at this scale. This "QK-Norm at scale" pattern was later adopted by several
frontier open LLMs (e.g., Gemma 2, and reportedly Qwen3 — see Section 3) as a standard
stability trick, not just a low-resource-translation trick as in the original 2020 paper.

### 2.2 Positional encoding

**Sinusoidal (original)** — from *Attention Is All You Need* itself, Section 3.5. Obstacle removed:
self-attention is permutation-invariant (no inherent notion of order), so position must be injected
explicitly; the authors chose a fixed (non-learned) function so the model could in principle
extrapolate to sequence lengths unseen during training (a hope only partially realized in practice —
see ALiBi/NoPE below).

$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}}), \quad PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

✅ **Su, Lu, Pan, Murtadha, Wen, Liu (2021), *RoFormer: Enhanced Transformer with Rotary Position
Embedding*, arXiv:2104.09864** (2021-04-20). Obstacle removed: absolute sinusoidal/learned PEs are
added once to the embedding and don't naturally encode *relative* position inside the attention dot
product itself. RoPE instead rotates each query/key vector by an angle proportional to its absolute
position, using pairwise 2D rotations; because rotation matrices compose, $Q_m \cdot K_n$ after
rotation depends only on the *relative* offset $(m-n)$ — unifying absolute and relative position
encoding with zero extra parameters. Now the de facto standard (GPT-NeoX, LLaMA, PaLM, Mistral,
DeepSeek, Qwen, …). For dimension pair $(2i, 2i{+}1)$: rotate by angle $m \cdot \theta_i$, where
$\theta_i = \text{base}^{-2i/d}$ (base commonly 10,000 originally, pushed to 500,000–1,000,000+ for
long-context models — see Section 4).

✅ **Press, Smith, Lewis (2021), *Train Short, Test Long: Attention with Linear Biases Enables Input
Length Extrapolation*, arXiv:2108.12409** (2021-08-27). Obstacle removed: models trained on short
sequences generally degrade sharply when evaluated on longer ones; instead of encoding position in
the embeddings at all, ALiBi adds a static, non-learned, distance-proportional penalty directly to
the pre-softmax attention scores (slope $m$ fixed per head), which extrapolates to much longer
sequences than sinusoidal/learned PE.

$$\text{softmax}(q_i k_j^\top + m\cdot(j-i))$$

✅ **Haviv, Ram, Press, Izsak, Levy (2022), *Transformer Language Models without Positional Encodings
Still Learn Positional Information*, arXiv:2203.16634** (2022-03-30, NoPE). Obstacle removed: showed
causal decoder-only LMs can implicitly recover positional information from the causal mask alone
(earlier tokens are structurally privileged) — competitive with explicit-PE models even with *zero*
positional encoding, questioning whether PE is even necessary in causal decoders.

✅ **Kazemnejad, Padhi, Ramamurthy, Das, Reddy (2023), *The Impact of Positional Encoding on Length
Generalization in Transformers*, arXiv:2305.19466** (2023-05-31). Obstacle removed: ran a controlled
empirical comparison and found NoPE actually generalizes to *longer* sequences than RoPE, ALiBi, or
learned absolute PE in decoder-only LMs — direct evidence against the "more explicit PE is always
better" assumption, and part of the motivation for interleaved RoPE/NoPE layer designs in some
2024–2025 long-context architectures (see Llama 4's reported "iRoPE" in Section 3).

**RoPE scaling for long context:**

✅ **Chen, Wong, Chen, Tian (2023), *Extending Context Window of Large Language Models via Positional
Interpolation*, arXiv:2306.15595** (2023-06-27). Obstacle removed: naively running RoPE past its
trained position range breaks the learned attention patterns (positions it never saw during
training); Position Interpolation instead linearly *downscales* position indices to stay within the
originally-trained range, extending LLaMA 7B–65B to 32,768 tokens with under 1,000 fine-tuning steps.

✅ **Peng, Quesnelle, Fan, Shippole (2023), *YaRN: Efficient Context Window Extension of Large
Language Models*, arXiv:2309.00071** (2023-08-31). Obstacle removed: plain Position Interpolation
treats all RoPE frequency dimensions identically, which is suboptimal (high-frequency dimensions
need less interpolation than low-frequency ones); YaRN applies NTK-aware, per-dimension
interpolation plus an attention-temperature correction, extending context 10x more token-efficiently
and with 2.5x fewer training steps than prior methods.

### 2.3 Activations / FFN

✅ **Shazeer (2020), *GLU Variants Improve Transformer*, arXiv:2002.05202** (2020-02-12). Obstacle
removed: the original Transformer FFN uses a plain ReLU/GELU nonlinearity between two matrices;
this paper shows gating the FFN with a Gated Linear Unit variant (SwiGLU, GEGLU, …) gives consistent
quality improvements at matched parameter/compute cost. SwiGLU is now the standard FFN in LLaMA,
PaLM, Mistral, DeepSeek, Qwen, and effectively every modern open LLM.

$$\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = \big(\text{Swish}_1(xW) \otimes xV\big) W_2, \quad \text{Swish}_1(x) = x \cdot \sigma(x) \ (\text{i.e. SiLU})$$

**Exact verbatim quote** on why the hidden dimension shrinks (pulled directly from the paper's PDF,
Section 2):

> "All of these layers have three weight matrices, as opposed to two for the original FFN. To keep
> the number of parameters and the amount of computation constant, we reduce the number of hidden
> units $d_{ff}$ (the second dimension of $W$ and $V$ and the first dimension of $W_2$) by a factor
> of $\frac{2}{3}$ when comparing these layers to the original two-matrix version."

Their own experiment's exact numbers (Section 3.1, also verified from the PDF): base model
$d_{model}=768$, $h=12$, $d_k=d_v=64$; ReLU baseline uses $d_{ff}=3072$ ($=4\times d_{model}$); the
GLU-variant FFNs use $d_{ff}=2048 = \frac{2}{3}\times 3072 = \frac{8}{3}\times 768$ — this is exactly
where the famous "8/3" convention comes from (see Section 4).

### 2.4 Attention variants

✅ **Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv:1911.02150**
(2019-11-06, MQA). Obstacle removed: autoregressive decoding is bottlenecked by memory bandwidth —
loading the growing KV cache at every step — not by FLOPs; Multi-Query Attention shares a *single*
K/V head across all query heads (only Q stays multi-head), shrinking the KV cache by a factor of $h$
and dramatically speeding up incremental decoding, at some cost to quality.

✅ **Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (2023), *GQA: Training Generalized
Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv:2305.13245** (2023-05-22).
Obstacle removed: MQA's single shared KV head can measurably hurt quality and training stability;
Grouped-Query Attention interpolates between full MHA and MQA by sharing K/V across *groups* of
query heads (e.g., 8 groups), keeping most of MQA's inference speed with much less quality loss —
and shows existing MHA checkpoints can be cheaply "uptrained" into GQA rather than retrained from
scratch. Now standard in LLaMA 2/3, Mistral, Qwen, and most dense open LLMs.

✅ **DeepSeek-AI et al. (2024), *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
Language Model*, arXiv:2405.04434** (2024-05-07, 157 authors, MLA). Obstacle removed: GQA/MQA reduce
the KV cache by literally sharing K/V heads across queries, which caps achievable quality; **Multi-head
Latent Attention (MLA)** instead low-rank-compresses the keys and values (and queries) of every token
into a small joint latent vector that is cached and decompressed on the fly, achieving KV-cache
savings comparable to or better than GQA/MQA while matching or exceeding full-MHA quality. 236B
total params, 21B activated per token, 128K context.

✅ **DeepSeek-AI et al. (2024), *DeepSeek-V3 Technical Report*, arXiv:2412.19437** (2024-12-27, 200
authors). Reuses and further validates MLA + DeepSeekMoE at much larger scale (671B total / 37B
activated params) and adds an auxiliary-loss-free load-balancing strategy plus a multi-token
prediction training objective (see Sections 2.5 and 3).

✅ **Beltagy, Peters, Cohan (2020), *Longformer: The Long-Document Transformer*, arXiv:2004.05150**
(2020-04-10). Obstacle removed: full self-attention scales $O(n^2)$ in sequence length, making
long-document processing prohibitively expensive; Longformer replaces it with local *sliding-window*
attention (linear in sequence length) plus a handful of task-specific global tokens that attend to
everything, as a drop-in replacement for standard self-attention.

✅ **Jiang, Sablayrolles, Mensch et al. (2023), *Mistral 7B*, arXiv:2310.06825** (2023-10-10, 18
authors). Obstacle removed: combines GQA (fast decoding) with sliding-window attention (each layer
only attends within a fixed local window), but stacked layers still let information propagate
across a much larger *effective* receptive field (analogous to dilated convolutions), reducing
inference cost for long sequences at production scale while outperforming larger dense models like
Llama 2 13B on most benchmarks.

### 2.5 Mixture of Experts

✅ **Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean (2017), *Outrageously Large Neural
Networks: The Sparsely-Gated Mixture-of-Experts Layer*, arXiv:1701.06538** (2017-01-23). Obstacle
removed: introduced a sparsely-gated MoE layer (noisy top-k gating between LSTM layers), showing
model *capacity* can scale far beyond what's affordable if every parameter had to be used on every
example — the "conditional computation" idea that underlies every MoE Transformer since.

✅ **Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen (2020), *GShard: Scaling Giant
Models with Conditional Computation and Automatic Sharding*, arXiv:2006.16668** (2020-06-30).
Obstacle removed: provided the systems/sharding annotation layer (plus top-2 gating with capacity
limits and an auxiliary load-balancing loss) needed to actually *train* multi-hundred-billion
parameter MoE Transformers across many accelerators — the systems backbone that made large-scale MoE
practical, not just theoretically possible.

✅ **Fedus, Zoph, Shazeer (2021), *Switch Transformers: Scaling to Trillion Parameter Models with
Simple and Efficient Sparsity*, arXiv:2101.03961** (2021-01-11). Obstacle removed: simplified MoE
routing to **top-1** (a single expert per token, "switch routing") instead of top-k≥2, cutting
communication/compute costs and training instability, enabling the first trillion-parameter sparse
models to train stably.

✅ **Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le, Laudon (2022), *Mixture-of-Experts with Expert
Choice Routing*, arXiv:2202.09368** (2022-02-18). Obstacle removed: standard token-choice routing
(each token picks its top-k experts) causes load imbalance — some experts get overloaded, others
under-trained; Expert Choice flips the direction so each *expert* selects its top tokens up to a
fixed buffer capacity, guaranteeing near-perfect load balance by construction with no auxiliary loss
needed (at the cost of complicating strictly-causal/online inference).

✅ **Dai, Deng, Zhao, Xu, Gao, Chen, Li, Zeng, Yu, Wu, Xie, Li, Huang, Luo, Ruan, Sui, Liang (2024),
*DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*,
arXiv:2401.06066** (2024-01-11). Obstacle removed: conventional top-K MoE (GShard/Switch-style)
wastes capacity because "knowledge" isn't cleanly separated across a small number of large experts;
DeepSeekMoE combines (a) very **fine-grained expert segmentation** (many small experts instead of
few large ones, for finer routing combinations) with (b) a subset of **shared experts** that are
always active for common knowledge, increasing specialization and reducing redundancy.

✅ **Wang, Gao, Zhao, Sun, Dai (2024), *Auxiliary-Loss-Free Load Balancing Strategy for
Mixture-of-Experts*, arXiv:2408.15664** (2024-08-28). Obstacle removed: standard auxiliary
load-balancing losses (Switch/GShard-style) directly interfere with the main language-modeling
gradient, hurting quality when weighted strongly enough to actually balance load; this method
instead adds a per-expert **bias term** to the routing decision only (not the loss), dynamically
adjusted after each training step based on observed load, achieving balance "for free" without
polluting the LM gradient. Adopted in DeepSeek-V3.

---

## 3. Current SOTA (2025 → mid-2026) and the live architecture debate

All entries below are ✅ arXiv-verified unless flagged otherwise.

| Model / paper | arXiv id | Date | Architecture notes |
|---|---|---|---|
| Qwen2.5 Technical Report (Qwen Team) | 2412.15115 | 2024-12-19 | Dense + MoE variants, GQA, RoPE; scaled pretraining to 18T tokens |
| **Llama 3 Herd of Models** (Grattafiori et al., 561 authors) | 2407.21783 | 2024-07-31 | Dense (no MoE) decoder-only; GQA (8 KV heads at every scale); RoPE θ=500,000; SwiGLU; 128K vocab |
| **Llama 4** | — | — | ❌ **No official Meta arXiv paper found.** The only arXiv record for "Llama 4" (2601.11659) was **removed by arXiv administrators** — comment field states: *"This version has been removed by arXiv administrators due to incorrect authorship. The name of the submitter was also incorrect."* It was an unauthorized third-party writeup, not a Meta publication. Publicly (per Meta's own blog/model card, **not arXiv-verified**), Llama 4 Scout/Maverick reportedly use MoE + early-fusion multimodality + "iRoPE" (interleaved RoPE/no-PE layers for length generalization) |
| DeepSeek-V2 | 2405.04434 | 2024-05-07 | MLA + DeepSeekMoE; 236B total / 21B active |
| DeepSeek-V3 Technical Report | 2412.19437 | 2024-12-27 | MLA + DeepSeekMoE; 671B total / 37B active; aux-loss-free balancing; multi-token prediction objective |
| DeepSeek-R1 | 2501.12948 | 2025-01-22 | Same MLA+MoE backbone as V3; reasoning emerges from large-scale RL, not a new base architecture |
| MiniMax-01 | 2501.08313 | 2025-01-14 | Hybrid: "Lightning Attention" (linear-attention variant) in most layers + periodic full softmax attention layers; targets ultra-long context |
| Native Sparse Attention (NSA) | 2502.11089 | 2025-02-16 | DeepSeek team; hardware-aligned, *natively trainable* sparse attention (sparsity baked in from pretraining, not bolted on at inference) |
| MoBA (Mixture of Block Attention) | 2502.13189 | 2025-02-18 | Moonshot/Kimi team; applies MoE-style gating to attention itself — each query dynamically routes to a sparse subset of KV "blocks" instead of a fixed local window |
| Qwen3 Technical Report | 2505.09388 | 2025-05-14 | Dense + MoE (0.6B–235B); unified "thinking"/"non-thinking" mode switch; GQA + RoPE |
| Kimi K2: Open Agentic Intelligence | 2507.20534 | 2025-07-28 | ~1T-parameter sparse MoE, MLA-style attention, agentic/tool-use focus |
| gpt-oss-120b & gpt-oss-20b Model Card (OpenAI) | 2508.10925 | 2025-08-08 | OpenAI's first open-weight models since GPT-2; sparse MoE |
| Kimi Linear: An Expressive, Efficient Attention Architecture | 2510.26692 | 2025-10-30 | Hybrid linear attention ("Kimi Delta Attention") interleaved with periodic full attention, matching full-attention quality with a smaller KV cache |
| OLMo 3 | 2512.13961 | 2025-12-15 | AI2's latest fully-open (data+code+weights+logs) model family |
| DeepSeek-V3.2 | 2512.02556 | 2025-12-02 | Latest DeepSeek iteration as of this survey; pushes the MLA+MoE recipe further (architecture deltas beyond V3 not deep-dived here — flagged for follow-up) |

Also relevant background (all ✅ verified): **OLMo** (2402.00838, Groeneveld et al., 2024-02-01) — the
original fully-open baseline; **2 OLMo 2 Furious** (2501.00656, Team OLMo, 2024-12-31) — fixed
training-stability issues (QK-norm + reordered norm placement, z-loss) to make fully-open training
competitive with Llama 3.1/Qwen2.5.

### The SSM / linear-attention / hybrid thread

✅ **Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*,
arXiv:2312.00752** (2023-12-01). Obstacle removed: prior structured state-space models (S4, etc.)
had time-*invariant* dynamics, limiting content-based reasoning (they can't selectively remember or
forget based on the input); Mamba makes the SSM's parameters input-dependent ("selective"),
enabling content-aware sequence modeling with linear-time (not quadratic) scaling — the first pure
SSM to match Transformer quality at equivalent scale on language modeling.

✅ **Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through
Structured State Space Duality*, arXiv:2405.21060** (2024-05-31, Mamba-2). Obstacle removed:
established a formal mathematical duality between SSMs and a structured form of linear attention,
unifying the two families theoretically and yielding a much faster hardware-efficient algorithm
("SSD") than the original Mamba's implementation.

✅ **De, Smith, Fernando et al. (2024), *Griffin: Mixing Gated Linear Recurrences with Local
Attention for Efficient Language Models*, arXiv:2402.19427** (2024-02-29, Google DeepMind). Obstacle
removed: showed a hybrid of gated linear recurrence (RG-LRU) + local sliding-window attention
matches Transformer quality at equal scale while giving much faster inference and lower memory for
long sequences than a pure Transformer (powers "RecurrentGemma").

✅ **Lieber, Lenz, Bata et al. (2024), *Jamba: A Hybrid Transformer-Mamba Language Model*,
arXiv:2403.19887** (2024-03-28, AI21 Labs). Obstacle removed: first large-scale (52B total / 12B
active) production-grade model to interleave Transformer and Mamba layers *plus* MoE in one
architecture, empirically demonstrating hybrid SSM-attention models can match Transformer quality
with substantially better throughput/memory at long context than pure attention.

### What is genuinely unresolved (synthesis across the above, not a single citation)

- **Pure attention vs. pure SSM vs. hybrid.** As of mid-2026, essentially every frontier lab that has
  moved away from pure dense/MoE attention (MiniMax-01, Jamba, Griffin, Kimi Linear) has landed on a
  **hybrid** — some full-attention layers interleaved with linear/SSM layers — rather than a pure SSM.
  No lab has shipped a pure-SSM frontier model at GPT-4/DeepSeek-V3 scale. This suggests full
  attention still has an edge on tasks like in-context copying/retrieval that pure SSMs are known to
  struggle with, but hybrids are one of the most active open research directions and the right mixing
  ratio/layer pattern is unresolved.
- **Sparse-attention-as-first-class-citizen (NSA, MoBA) is a *third* camp**, distinct from both dense
  attention and SSMs: keep the quadratic attention mechanism's expressiveness but make it sparse and
  *natively trainable* (rather than only pruned at inference). Both NSA (DeepSeek) and MoBA
  (Moonshot/Kimi) appeared within days of each other in February 2025 — a live, unresolved
  three-way competition between "replace attention" (SSM/linear), "make attention sparse" (NSA/MoBA),
  and "keep dense attention, just compress the KV cache" (MLA/GQA).
- **MLA vs. GQA.** DeepSeek's MLA claims to beat GQA on the quality/KV-cache-size tradeoff, and Kimi
  K2 has adopted an MLA-style attention too, but most Western labs surveyed here (Meta/Llama, Mistral,
  OpenAI's gpt-oss) still default to GQA or sliding-window attention rather than MLA. Which approach
  becomes the long-term standard is genuinely unsettled.
- **Is explicit positional encoding even necessary?** The NoPE evidence (Haviv et al.,
  Kazemnejad et al.) suggests causal decoders don't strictly need it and may generalize to length
  *better* without it, yet the field's default is still RoPE with various scaling tricks. Llama 4's
  reported "iRoPE" (interleaving RoPE and no-PE layers) is one attempted middle ground — but notably,
  the only arXiv writeup of Llama 4's architecture was pulled by arXiv administrators for authorship
  problems, underscoring how undocumented/unsettled some of the very newest design choices are.
- **Auxiliary-loss-free load balancing** (DeepSeek-V3) is a clear improvement over gradient-polluting
  auxiliary losses within DeepSeek's own ablations, but it has not yet been widely validated by other
  labs' from-scratch training runs at comparable scale — still an open question whether it generalizes.

---

## 4. Quiz-ready numbers and formulas

1. **Scaled dot-product attention & the 1/√d_k factor** (Vaswani et al., arXiv:1706.03762):
   $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$. If Q, K components are i.i.d.
   mean-0 variance-1, the dot product $q\cdot k$ has variance $d_k$ — scaling by $1/\sqrt{d_k}$
   renormalizes the logits back to unit variance, preventing softmax saturation into
   near-zero-gradient regions. Base model: $d_{model}{=}512$, $h{=}8$, $d_k{=}d_v{=}64$.

2. **Decoder-only Transformer block parameter-count approximation** (standard formula, e.g. Kaplan
   et al. scaling-laws convention): per layer, self-attention contributes $4 d_{model}^2$
   ($W_Q,W_K,W_V,W_O$, each $d_{model}\times d_{model}$) and a standard (non-gated, $4\times$
   expansion) FFN contributes $8 d_{model}^2$ ($W_1$: $d_{model}\times 4d_{model}$, $W_2$:
   $4d_{model}\times d_{model}$) → **≈ $12\,d_{model}^2$ per layer**, so a full model is
   $N \approx 12 \cdot n_{layer} \cdot d_{model}^2$ (plus embedding params $\approx
   vocab\_size \times d_{model}$, often tied/excluded as a lower-order term).

3. **SwiGLU's "8/3" hidden-dim convention** (Shazeer, arXiv:2002.05202, verbatim + exact numbers
   verified from the paper): a gated FFN has **3** weight matrices instead of 2, so to match the
   baseline's parameter/compute count, $d_{ff}$ is shrunk by a factor of $\tfrac{2}{3}$:
   $d_{ff}^{GLU} = \tfrac{2}{3}\times(4\, d_{model}) = \tfrac{8}{3}\, d_{model}$. Verified worked
   example from the paper itself: $d_{model}{=}768 \Rightarrow$ baseline $d_{ff}{=}3072$, GLU-variant
   $d_{ff}{=}2048$ ($=\tfrac{8}{3}\times768$). (Real implementations like LLaMA further apply a
   custom multiplier and round up to a multiple of 256, so shipped $d_{ff}$ values deviate somewhat
   from the pure 8/3 ratio — e.g. Llama 3 8B uses $d_{ff}{=}14336 = 3.5\times d_{model}$, not
   $2.667\times$.)

4. **GQA group counts in Llama 3** (verified from arXiv:2407.21783's hyperparameter table): all three
   sizes use **8 key/value heads**, only the query-head count (and therefore the group size) scales
   up — 8B: 32 query heads / 8 KV heads (4:1 grouping); 70B: 64 query heads / 8 KV heads (8:1);
   405B: 128 query heads / 8 KV heads (16:1). All three also share the same **RoPE base
   $\theta = 500{,}000$** (up from the original RoFormer/GPT-NeoX default of $\theta=10{,}000$),
   supporting stable extrapolation to a 128K context after continued pretraining.

5. **RoPE base-theta scaling as a crude long-context lever**: original RoFormer/GPT-NeoX/LLaMA-1 use
   $\theta{=}10{,}000$; Llama 3 (and many 2024+ long-context models) use $\theta{=}500{,}000$; some
   long-context fine-tunes (e.g. Code Llama-style extensions) push to $\theta{=}1{,}000{,}000+$. Larger
   $\theta$ lowers the rotation frequency of every dimension, which slows the rate at which relative
   positional signal decays/aliases at long distances — the crude version of what YaRN
   (arXiv:2309.00071) and Position Interpolation (arXiv:2306.15595) do more carefully on a
   per-frequency-dimension basis.

6. **MoE sparsity ratios at the frontier** (verified from abstracts): DeepSeek-V2 = 236B total / 21B
   activated per token (~8.9% active); DeepSeek-V3 = 671B total / 37B activated (~5.5% active) —
   illustrating the general MoE principle that *total* capacity can scale far ahead of *active*
   (and therefore FLOPs-per-token) compute, which is the entire premise of Shazeer's 2017
   sparsely-gated MoE (arXiv:1701.06538).

---

## 5. Full paper index

All arXiv ids below were returned by an actual `ti:"<exact title>"` API query (see verification key
above). Papers marked ❌ were queried and returned zero results.

| # | Paper | First author | Year | arXiv id | Status |
|---|---|---|---|---|---|
| 1 | Attention Is All You Need | Vaswani | 2017 | 1706.03762 | ✅ |
| 2 | Neural Machine Translation by Jointly Learning to Align and Translate | Bahdanau | 2014 | 1409.0473 | ✅ |
| 3 | Sequence to Sequence Learning with Neural Networks | Sutskever | 2014 | 1409.3215 | ✅ |
| 4 | Improving Language Understanding by Generative Pre-Training (GPT-1) | Radford | 2018 | — | ❌ not on arXiv |
| 5 | Language Models are Unsupervised Multitask Learners (GPT-2) | Radford | 2019 | — | ❌ not on arXiv |
| 6 | Language Models are Few-Shot Learners (GPT-3) | Brown | 2020 | 2005.14165 | ✅ |
| 7 | BERT: Pre-training of Deep Bidirectional Transformers | Devlin | 2018 | 1810.04805 | ✅ |
| 8 | Layer Normalization | Ba | 2016 | 1607.06450 | ✅ |
| 9 | Root Mean Square Layer Normalization | Zhang | 2019 | 1910.07467 | ✅ |
| 10 | On Layer Normalization in the Transformer Architecture | Xiong | 2020 | 2002.04745 | ✅ |
| 11 | Query-Key Normalization for Transformers | Henry | 2020 | 2010.04245 | ✅ |
| 12 | Scaling Vision Transformers to 22 Billion Parameters | Dehghani | 2023 | 2302.05442 | ✅ |
| 13 | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su | 2021 | 2104.09864 | ✅ |
| 14 | Train Short, Test Long (ALiBi) | Press | 2021 | 2108.12409 | ✅ |
| 15 | Transformer LMs without Positional Encodings Still Learn Positional Info (NoPE) | Haviv | 2022 | 2203.16634 | ✅ |
| 16 | The Impact of Positional Encoding on Length Generalization | Kazemnejad | 2023 | 2305.19466 | ✅ |
| 17 | Extending Context Window via Positional Interpolation | Chen | 2023 | 2306.15595 | ✅ |
| 18 | YaRN: Efficient Context Window Extension | Peng | 2023 | 2309.00071 | ✅ |
| 19 | GLU Variants Improve Transformer | Shazeer | 2020 | 2002.05202 | ✅ |
| 20 | Fast Transformer Decoding: One Write-Head is All You Need (MQA) | Shazeer | 2019 | 1911.02150 | ✅ |
| 21 | GQA: Training Generalized Multi-Query Transformer Models | Ainslie | 2023 | 2305.13245 | ✅ |
| 22 | DeepSeek-V2 (MLA + DeepSeekMoE) | DeepSeek-AI | 2024 | 2405.04434 | ✅ |
| 23 | DeepSeek-V3 Technical Report | DeepSeek-AI | 2024 | 2412.19437 | ✅ |
| 24 | Longformer: The Long-Document Transformer | Beltagy | 2020 | 2004.05150 | ✅ |
| 25 | Mistral 7B | Jiang | 2023 | 2310.06825 | ✅ |
| 26 | Outrageously Large Neural Networks (sparsely-gated MoE) | Shazeer | 2017 | 1701.06538 | ✅ |
| 27 | Switch Transformers | Fedus | 2021 | 2101.03961 | ✅ |
| 28 | GShard | Lepikhin | 2020 | 2006.16668 | ✅ |
| 29 | DeepSeekMoE | Dai | 2024 | 2401.06066 | ✅ |
| 30 | Mixture-of-Experts with Expert Choice Routing | Zhou | 2022 | 2202.09368 | ✅ |
| 31 | Auxiliary-Loss-Free Load Balancing Strategy for MoE | Wang | 2024 | 2408.15664 | ✅ |
| 32 | DeepSeek-R1 | DeepSeek-AI | 2025 | 2501.12948 | ✅ |
| 33 | Qwen3 Technical Report | Yang | 2025 | 2505.09388 | ✅ |
| 34 | Qwen2.5 Technical Report | Qwen Team | 2024 | 2412.15115 | ✅ |
| 35 | The Llama 3 Herd of Models | Grattafiori | 2024 | 2407.21783 | ✅ |
| 36 | Llama 4 (official Meta paper) | — | — | — | ❌ no legitimate arXiv paper (only record was arXiv-admin-redacted for incorrect authorship) |
| 37 | OLMo: Accelerating the Science of Language Models | Groeneveld | 2024 | 2402.00838 | ✅ |
| 38 | 2 OLMo 2 Furious | Team OLMo | 2024/25 | 2501.00656 | ✅ |
| 39 | Olmo 3 | Team Olmo | 2025 | 2512.13961 | ✅ |
| 40 | Kimi K2: Open Agentic Intelligence | Kimi Team | 2025 | 2507.20534 | ✅ |
| 41 | Kimi Linear: An Expressive, Efficient Attention Architecture | Kimi Team | 2025 | 2510.26692 | ✅ |
| 42 | MiniMax-01: Scaling Foundation Models with Lightning Attention | MiniMax | 2025 | 2501.08313 | ✅ |
| 43 | gpt-oss-120b & gpt-oss-20b Model Card | OpenAI | 2025 | 2508.10925 | ✅ |
| 44 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu | 2023 | 2312.00752 | ✅ |
| 45 | Transformers are SSMs (Mamba-2 / SSD) | Dao | 2024 | 2405.21060 | ✅ |
| 46 | Jamba: A Hybrid Transformer-Mamba Language Model | Lieber | 2024 | 2403.19887 | ✅ |
| 47 | Griffin: Mixing Gated Linear Recurrences with Local Attention | De | 2024 | 2402.19427 | ✅ |
| 48 | Native Sparse Attention (NSA) | Yuan | 2025 | 2502.11089 | ✅ |
| 49 | MoBA: Mixture of Block Attention | Lu | 2025 | 2502.13189 | ✅ |
| 50 | DeepSeek-V3.2 | DeepSeek-AI | 2025 | 2512.02556 | ✅ |

Queries attempted but not needed in the final writeup (found nothing / superseded by a better match):
"Qwen3-Next" (❌ not found on arXiv under that exact title or in abstracts — likely a blog-only
release without a dedicated arXiv technical report).

Not independently investigated in depth for this pass (flagged for potential follow-up if the
explainer needs more current-model detail): DeepSeek-V3.2's specific architectural deltas beyond V3;
gpt-oss's exact attention-layer pattern (reported to alternate sliding-window and full attention,
per the model card, but not verified line-by-line from the PDF in this pass).
