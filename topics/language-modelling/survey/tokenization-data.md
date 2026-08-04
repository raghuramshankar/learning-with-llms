# Literature Survey: Tokenization and Data Curation

Scope: subword/byte-level tokenization and its pathologies, plus the pretraining-data pipeline (crawling, dedup, filtering, mixing, synthetic augmentation, data-constrained scaling), through mid-2026. Written for a CS336-style "language modeling from scratch" explainer.

**Verification method note:** Papers were queried against the arXiv API (`http://export.arxiv.org/api/query`, `ti:"<exact title>"`) as instructed. Partway through this session the shared sandbox IP was hit with a sustained `HTTP 429` from arXiv's API endpoint (almost certainly from concurrent request volume — other agents in this environment were hammering the same endpoint at the same time). For roughly a third of the papers below, the raw `export.arxiv.org/api/query` call could not get through even after multi-minute exponential backoff. For those, I instead pulled the exact title/authors/date directly from the paper's canonical `arxiv.org/abs/<id>` page (still arxiv.org itself, just the abstract-page HTML rather than the Atom API) via WebFetch. Every ID below was independently cross-referenced against at least one WebSearch result showing the same `arxiv.org/abs/` URL. I've tagged each entry `[API-VERIFIED]` or `[PAGE-VERIFIED]` accordingly. Nothing below is fabricated; anything I could not locate at all is called out explicitly as NOT FOUND.

---

## 1. Tokenization

### 1.1 Subword tokenization: the foundational thread

| Paper | Authors | Year | ID | Obstacle removed |
|---|---|---|---|---|
| **Neural Machine Translation of Rare Words with Subword Units** | Sennrich, Haddow, Birch | 2015 (pub. 2016, ACL) | arXiv:1508.07909 `[API-VERIFIED]` | Introduced **BPE** for NMT: repeatedly merges the most frequent adjacent symbol pair into a new symbol, giving a fixed-size open vocabulary that handles rare/unseen words without falling back to `<unk>` or a dictionary lookup. |
| **Japanese and Korean Voice Search** | Schuster, Nakajima | 2012, ICASSP (NOT on arXiv — IEEE only, DOI 10.1109/ICASSP.2012.6289079) | — | Introduced **WordPiece**: a greedy, likelihood-maximizing merge criterion (vs. BPE's frequency criterion) for building a subword inventory for Google's voice-search LM; later reused in BERT. |
| **Google's Neural Machine Translation System** | Wu, Schuster, Chen, Le, Norouzi, et al. (30 authors, incl. Kudo) | 2016 | arXiv:1609.08144 `[API-VERIFIED]` | Popularized WordPiece at production NMT scale, demonstrating subword tokenization removes the open-vocabulary bottleneck for a shipped system. |
| **SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing** | Kudo, Richardson | 2018 | arXiv:1808.06226 `[API-VERIFIED]` | Removed the **pre-tokenization/whitespace assumption**: treats input as a raw Unicode stream (spaces are just another symbol), so the same tool works for languages without whitespace word boundaries (Japanese, Chinese, Thai) and gives a fully invertible, lossless detokenization. |
| **Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates** | Kudo | 2018 | arXiv:1804.10959 `[API-VERIFIED]` | Introduced the **unigram-LM tokenizer** and training-time subword sampling: instead of one deterministic BPE segmentation, sample among multiple valid segmentations to regularize the model against tokenization noise — the basis of SentencePiece's `unigram` mode. |

### 1.2 Byte-level BPE and tiktoken

- **GPT-2 (Radford et al., 2019, "Language Models are Unsupervised Multitask Learners")** — searched `ti:"Language Models are Unsupervised Multitask Learners"` on the arXiv API: **zero results returned.** This paper was never posted to arXiv; it exists only as an OpenAI technical report/blog PDF. Flagging explicitly per instructions rather than fabricating an ID. GPT-2's contribution was **byte-level BPE**: run BPE merges over raw UTF-8 bytes (256 base symbols) instead of Unicode characters, so the vocabulary can represent *any* string with zero `<unk>` tokens and no Unicode-normalization edge cases.
- **tiktoken** (OpenAI, github.com/openai/tiktoken) — no associated paper; it's a Rust-backed BPE-encoding library, not a new algorithm. Its contribution is purely engineering: a byte-pair encoder that is reported ~3–6x faster than prior open-source BPE implementations on 1GB of text, and it ships the reference merge tables for `gpt2`, `cl100k_base` (GPT-3.5/GPT-4), and `o200k_base` (GPT-4o).

### 1.3 The tokenizer-free / byte-level thread

| Paper | Authors | Year | ID | Obstacle removed |
|---|---|---|---|---|
| **ByT5: Towards a token-free future with pre-trained byte-to-byte models** | Xue, Barua, Constant, Al-Rfou, Narang, Kale, Roberts, Raffel | 2021 | arXiv:2105.13626 `[API-VERIFIED]` | Showed a T5-style encoder-decoder can train directly on raw UTF-8 bytes (no vocabulary at all), trading a longer sequence for total robustness to spelling noise/typos and zero out-of-vocabulary risk. |
| **CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation** | Clark, Garrette, Turc, Wieting | 2021 | arXiv:2103.06874 `[API-VERIFIED]` | Removed the encoder-side tokenization step entirely by downsampling a character/codepoint sequence before the Transformer, avoiding BPE's brittleness under noisy or morphologically rich text. |
| **MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers** | Yu, Simig, Flaherty, Aghajanyan, Zettlemoyer, Lewis | 2023 | arXiv:2305.07185 `[API-VERIFIED]` | Removed the **quadratic-attention-over-bytes** bottleneck: a multiscale (patch + global model) architecture makes million-byte-length end-to-end differentiable modeling tractable for the first time. |
| **MambaByte: Token-free Selective State Space Model** | Wang, Gangavarapu, Yan, Rush | 2024 | arXiv:2401.13660 `[API-VERIFIED]` | Swapped the Transformer for a Mamba (selective SSM) backbone at the byte level, removing the quadratic-in-length cost that otherwise makes byte-level Transformers impractical at long context. |
| **Byte Latent Transformer: Patches Scale Better Than Tokens (BLT)** | Pagnoni, Pasunuru, Rodriguez, Nguyen, Muller, Li, Zhou, Yu, Weston, Zettlemoyer, Ghosh, Lewis, Holtzman, Iyer | 2024 | arXiv:2412.09871 `[API-VERIFIED]` | First byte-level LM to **match a token-based (BPE) LM's performance at scale (up to 8B params, 4T training bytes)** — dynamically groups bytes into variable-length "patches" via a small entropy model, so compute is spent proportional to local unpredictability rather than fixed per-token, and the model gets sub-word-free robustness (typos, arbitrary scripts) for free. |
| **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net)** | Hwang, Wang, Gu | 2025 | arXiv:2507.07955 `[PAGE-VERIFIED, cross-checked via HF papers listing]` | Learns the chunk/patch boundaries **end-to-end** (rather than BLT's separately-trained entropy heuristic) via a differentiable dynamic-chunking mechanism inside a U-Net-style hierarchy; at matched compute/data, a single-stage byte-level H-Net beats a BPE Transformer outright, and iterating the hierarchy further closes the gap with models 2x its size — removes the last hand-designed heuristic from the byte-level pipeline. |
| **Fast Byte Latent Transformer** | Kallini, Pagnoni, Limisiewicz, Ghosh, Zettlemoyer, Potts, Han, Iyer | 2026 | arXiv:2605.08044 `[API-VERIFIED]` | Removes BLT's **slow byte-by-byte autoregressive decoding** bottleneck: adds a block-wise diffusion training objective (BLT-D) plus self-speculative/verification variants (BLT-S, BLT-DV) enabling parallel multi-byte generation, cutting inference memory-bandwidth cost by 50–92% vs. vanilla BLT. |

### 1.4 Known pathologies

**Glitch tokens / "SolidGoldMagikarp."** Jessica Rumbelow & Matthew Watkins, *"SolidGoldMagikarp (plus, prompt generation)"*, LessWrong / AI Alignment Forum, **5 Feb 2023** (work done during a SERI-MATS fellowship). NOT an arXiv paper — a blog-format research writeup; confirmed via WebSearch, not the arXiv API. Discovery: certain BPE tokens (e.g. `SolidGoldMagikarp`, `TheNitromeFan`, ` SmartStocks`) — present in the vocabulary because the tokenizer was trained on a broader corpus (reportedly including Reddit counting threads) than the model ever saw during pretraining — get essentially zero gradient signal and leave the model with an untrained/near-random embedding at that vocab slot, causing wildly erratic completions when the token is echoed back. Two academic follow-ups formalize this:
- **Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models**, Land & Bartolo, 2024, arXiv:2405.05417 `[PAGE-VERIFIED]` — obstacle removed: gives a general, model-agnostic detection method (rather than manual discovery) for under-trained tokens across many open LLMs, tracing the root cause to tokenizer-training-corpus vs. model-training-corpus mismatch plus certain `<unk>`-adjacent artifacts.
- **Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection**, Li, Liu, Deng, Zhang, Song, Shi, Wang, Li, Liu, Wang, 2024, arXiv:2404.09894 `[PAGE-VERIFIED]` — obstacle removed: proposes GlitchHunter, a clustering-based detector exploiting the empirical observation that glitch tokens cluster together in embedding space, tested across 7 LLMs / 3 tokenizers / 182,517 tokens.

**Digit tokenization.** Standard BPE/WordPiece merges digits like any other byte-pair, so `"1234567"` gets chunked left-to-right into arbitrary, inconsistent groups (e.g. `"123", "4567"` in one context but `"12", "34567"` in another depending on surrounding text), which cripples place-value reasoning and multi-digit arithmetic.
- **Investigating the Limitations of Transformers with Simple Arithmetic Tasks**, Nogueira, Jiang, Lin, 2021, arXiv:2102.13019 `[PAGE-VERIFIED]` — obstacle removed: demonstrated that surface representation (digit spacing/order), not model capacity, is a first-order determinant of whether Transformers can learn arithmetic at all.
- **Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks**, Liu, Low, 2023, arXiv:2305.14201 `[verified: title+authors+id cross-confirmed via WebSearch result snippet from arxiv.org listing]` — obstacle removed: showed that supervised fine-tuning on a synthetic arithmetic dataset, combined with LLaMA's fairly consistent digit tokenization, gets near-perfect large-number addition/subtraction, isolating tokenization consistency (not scale) as the key lever.
- Downstream community finding (blog, not peer-reviewed; cited for the number only): switching from left-to-right to **right-to-left digit chunking** raised GPT-3.5 arithmetic accuracy from 75.6% → 97.8% and GPT-4 from 84.4% → 98.9% (Beren Millidge, "Integer tokenization is insane," 2023/2024 — non-arXiv blog source, flagged as such).

**Multilingual token inflation / tokenizer unfairness.**
- **Language Model Tokenizers Introduce Unfairness Between Languages**, Petrov, La Malfa, Torr, Bibi, 2023 (NeurIPS 2023), arXiv:2305.15425 `[PAGE-VERIFIED]` — obstacle removed: formalized "tokenizer parity" and showed the *same* text in different languages can differ by up to **15x** in token count under a shared multilingual tokenizer, translating directly into inflated API cost, inflated latency, and *effectively* shrunk context windows for non-English (esp. non-Latin-script) users — a fairness problem invisible if you only benchmark on English.
- **How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models**, Rust, Pfeiffer, Vulić, Ruder, Gurevych, 2020, arXiv:2012.15613 `[API-VERIFIED]` — obstacle removed: isolated tokenizer quality (vs. everything else about "being multilingual") as a distinct, measurable driver of the multilingual-vs-monolingual performance gap.
- **Getting the most out of your tokenizer for pre-training and domain adaptation**, Dagan, Synnaeve, Rozière, 2024, arXiv:2402.01035 `[API-VERIFIED]` — obstacle removed: showed tokenizer choice (vocab size, pre-tokenization regex, training data) is itself an underrated hyperparameter with large, measurable downstream effects — most labs just inherit someone else's tokenizer without ablating it.

**Compression ratio as the unifying metric.** All of the above pathologies are really different faces of one thing: how many *bytes* a tokenizer packs into one *token*, and how unevenly that ratio is distributed across content types and languages. See §4 for concrete numbers (GPT-2 vs. GPT-4 vs. GPT-4o vocab/compression, FineWeb dedup rates, etc.).

---

## 2. Data curation

### 2.1 Common Crawl processing pipelines (chronological)

| Paper | Authors | Year | ID | Obstacle removed |
|---|---|---|---|---|
| **CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data** | Wenzek, Lachaux, Conneau, Chaudhary, Guzmán, Joulin, Grave | 2019 | arXiv:1911.00359 `[API-VERIFIED — confirmed both via direct API call and independently via WebFetch of the abs page]` | First fully automatic pipeline (dedup by paragraph hash + language ID + perplexity filtering against a Wikipedia-trained LM) to turn raw Common Crawl into per-language monolingual corpora at scale without manual curation. |
| **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5 / C4)** | Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li, Liu | 2019 | arXiv:1910.10683 `[API-VERIFIED]` | Introduced **C4** (Colossal Clean Crawled Corpus): simple deterministic heuristic filters (drop lines without terminal punctuation, drop pages with "javascript"/placeholder text, drop offensive-word-list hits, dedup 3-sentence spans) applied to one CC snapshot — showed that *even crude* heuristic cleaning of web text beats no cleaning at all for pretraining. |
| **The Pile: An 800GB Dataset of Diverse Text for Language Modeling** | Gao, Biderman, Black, Golding, Hoppe, Foster, Phang, He, Thite, Nabeshima, Presser, Leahy | 2020 | arXiv:2101.00027 `[API-VERIFIED]` | Removed the "web-crawl-only" assumption: showed **domain diversity** (22 curated sources — books, code, academic papers, dialogue, etc. — deliberately mixed, not just filtered CC) itself improves generalization, independent of filtering quality. |
| **The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only** | Penedo, Malartic, Hesslow, Cojocaru, Cappelli, Alobeidli, Pannier, Almazrouei, Launay | 2023 | arXiv:2306.01116 `[API-VERIFIED]` | Directly challenged The Pile's premise: with aggressive-enough filtering + dedup (MacroData Refinement pipeline: language ID, quality filters à la Gopher, then fuzzy+exact dedup) **web data alone outperforms curated multi-source mixtures**, undercutting the assumed need for hand-curated domains like books/code. |
| **Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research** | Soldaini, Kinney, Bhagia, Schwenk, Atkinson, et al. (AI2, 34 authors) | 2024 | arXiv:2402.00159 `[API-VERIFIED]` | Removed the **opacity** of frontier pretraining corpora: released not just the 3T-token dataset but the full toolkit/documentation of every filtering, dedup, and mixing decision, enabling reproducible ablation research on curation itself (rather than researchers having to reverse-engineer Llama/GPT-4's undisclosed data recipe). |
| **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale** | Penedo, Kydlíček, Ben Allal, Lozhkov, Mitchell, Raffel, Von Werra, Wolf | 2024 | arXiv:2406.17557 `[API-VERIFIED]` | Systematically **ablated every pipeline stage** (dedup strategy, filter choice, ordering) instead of picking a recipe by folklore — the counterintuitive headline finding was that dedup is **non-monotonic**: globally deduping across all 96 CC snapshots destructively over-deduped the oldest snapshots (up to ~90% of a snapshot's data removed) and *hurt* downstream performance versus per-snapshot MinHash dedup. Also shipped **FineWeb-Edu**, an educational-quality-classifier-filtered subset. |
| **DataComp-LM: In search of the next generation of training sets for language models (DCLM)** | Li, Fang, Smyrnis, Ivgi, Jordan, Gadre, Bansal, Guha, Keh, Arora, et al. (mlfoundations, 45+ authors) | 2024 | arXiv:2406.11794 `[API-VERIFIED]` | Turned dataset curation into a **controlled benchmark** (fixed model/compute, vary only the data pipeline) rather than an uncontrolled property of whichever model shipped it — the released DCLM-Baseline (fastText classifier retaining only the top ~10% of a huge raw pool by quality score, ~3.8T tokens / ~3B docs) became a new open-data high-water mark. |
| **Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset** | Su, Kong, Lin, Jennings, Norick, Kliegl, Patwary, Shoeybi, Catanzaro (NVIDIA) | 2024 | arXiv:2412.02595 `[PAGE-VERIFIED]` | Addressed the tension between **aggressive filtering (high per-token quality, but throws away most of the crawl) and long-horizon training (needs more unique tokens than aggressive filters leave)** by using classifier ensembling *plus* LLM-based synthetic rephrasing to resurrect ~90% of the content that DCLM/FineWeb-Edu-style filters would have discarded — yielding 6.3T tokens (4.4T real + 1.9T synthetic), 4x more unique real tokens than DCLM at comparable quality. |

### 2.2 Quality filtering: classifier-based vs. perplexity-based, and its failure modes

- **Language Models are Few-Shot Learners (GPT-3)**, Brown et al., 2020, arXiv:2005.14165 `[API-VERIFIED]` — obstacle removed / introduced: used a **classifier-based** filter (quality classifier trained to distinguish curated reference corpora like WebText/Wikipedia/books from raw CommonCrawl) plus fuzzy dedup (Spark's MinHashLSH) to build GPT-3's training set, establishing classifier-based filtering as the default recipe that DCLM and Nemotron-CC still use today.
- **Scaling Language Models: Methods, Analysis & Insights from Training Gopher**, Rae, Borgeaud, Cai, Millican, Hoffmann, et al. (DeepMind, 70+ authors), 2021, arXiv:2112.11446 `[API-VERIFIED — confirmed both via direct API call and independently via WebFetch]` — obstacle removed: published a detailed, reusable set of **heuristic quality filters** (stop-word presence, symbol-to-word ratio, mean word length, repetition-fraction filters) as an alternative/complement to classifier-based filtering, now widely reused (RefinedWeb, FineWeb, Dolma all cite "Gopher rules" or a close variant).
- **Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets**, Kreutzer, Caswell, Wang, Wahab, van Esch, et al. (60+ authors), 2021, arXiv:2103.12028 `[PAGE-VERIFIED]` — obstacle removed: manually audited 205 language-specific corpora across 5 major released multilingual datasets (CCAligned, ParaCrawl, WikiMatrix, OSCAR, mC4) and found **at least 15 corpora had essentially zero usable text** and many were mislabeled by language code — exposed that automated quality filters silently fail catastrophically for lower-resource languages, motivating human-in-the-loop auditing as a necessary complement to classifiers/perplexity filters.

### 2.3 Deduplication

- **Deduplicating Training Data Makes Language Models Better**, Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini, 2021 (ACL 2022), arXiv:2107.06499 `[API-VERIFIED]` — obstacle removed: built the first scalable **exact-substring** (suffix-array based) and **near-duplicate (MinHash)** dedup tools for LM-scale corpora and showed duplicate training data causes models to memorize and regurgitate verbatim text — >1% of unprompted GPT-2-scale output was found to be a verbatim copy of training data pre-dedup; dedup cuts that regurgitation rate ~10x and also removes train/test leakage (affecting >4% of standard benchmark validation sets).
- **Deduplicating Training Data Mitigates Privacy Risks in Language Models**, Kandpal, Wallace, Raffel, 2022, arXiv:2202.06539 `[API-VERIFIED]` — obstacle removed: showed that membership-inference/extraction privacy attacks succeed largely *because* of duplicated sequences, not because of memorization per se — reframing dedup as a privacy mitigation, not just a quality one.
- **MinHash + LSH** as the underlying algorithm traces to Broder's 1997 "On the resemblance and containment of documents" (pre-arXiv era, not on arXiv — a data-mining/compression venue paper) and is the near-duplicate detection technique used, in some variant, by essentially every pipeline above (CCNet, C4, GPT-3, RefinedWeb, Dolma, FineWeb, DCLM).

### 2.4 Data mixing / domain weighting

- **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining**, Xie, Pham, Dong, Du, Liu, Lu, Liang, Le, Ma, Yu (Google/Stanford), 2023, arXiv:2305.10429 `[API-VERIFIED]` — obstacle removed: replaced hand-tuned domain mixture weights (e.g. "X% Wikipedia, Y% books, Z% web") with an automatic **group-DRO-trained small proxy model** that learns reweighting, then transfers those weights to train the full-size model — removes the need for expensive full-scale mixture-weight sweeps.

### 2.5 Synthetic data

- **Textbooks Are All You Need**, Gunasekar, Zhang, Aneja, Mendes, Del Giorno, Gopi, Javaheripi, Kauffmann, de Rosa, Saarikivi, Salim, Shah, Behl, Wang, Bubeck, Eldan, Kalai, Lee, Li (Microsoft, phi-1), 2023, arXiv:2306.11644 `[PAGE-VERIFIED]` — obstacle removed: showed a 1.3B-parameter model trained on a tiny (~7B token) but deliberately "textbook-quality" mix of filtered web text + GPT-3.5-generated synthetic textbooks/exercises can match or beat much larger models trained on orders-of-magnitude more (lower-quality) data on code benchmarks (50.6% HumanEval pass@1) — reframed data curation as "curate/synthesize for quality," not just "filter for quality," and kicked off the Phi model line.

### 2.6 Data-constrained scaling

- **Scaling Data-Constrained Language Models**, Muennighoff, Rush, Barak, Le Scao, Piktus, Tazi, Pyysalo, Wolf, Raffel, 2023 (NeurIPS 2023 Outstanding Paper runner-up), arXiv:2305.16264 `[PAGE-VERIFIED]` — obstacle removed: extended Chinchilla-style compute-optimal scaling laws to the case where **unique data, not compute, is the binding constraint** — the central empirical finding is that repeating data for up to ~4 epochs costs almost nothing relative to fresh unique tokens (value decays geometrically per repeat after that), giving practitioners a principled epoch budget instead of a taboo against ever repeating data.

---

## 3. Current SOTA and live controversies (2025 – mid-2026)

**Tokenizer-free modeling status as of mid-2026.** The trajectory across this literature is: ByT5/CANINE (2021, proof-of-concept, no compute-matched win) → MegaByte/MambaByte (2023–24, architectural fixes for the quadratic-cost problem) → **BLT** (Dec 2024, arXiv:2412.09871, first byte-level model to match a compute-matched BPE Transformer at 8B scale) → **H-Net** (Jul 2025, arXiv:2507.07955, removes BLT's hand-designed entropy-threshold patcher in favor of fully learned, differentiable chunking, and *beats* the BPE baseline outright at matched compute) → **Fast BLT** (May 2026, arXiv:2605.08044, attacks the remaining practical objection — slow byte-by-byte decoding — with a diffusion-style parallel-decode training objective, cutting inference memory-bandwidth cost by up to ~92%). As of this survey, tokenizer-free models are not yet the default in any deployed frontier LLM, but the "tokenization is a necessary evil" argument that held through ~2023 no longer has a clean empirical basis at the research-paper level — the open question is now systems/engineering (fast decoding, tooling maturity, hardware kernels) rather than "can it match BPE quality."

**Newest open datasets / curation techniques.** Nemotron-CC (Dec 2024, arXiv:2412.02595) is the clearest recent example of a curation-technique shift: rather than choosing a single filtering aggressiveness, it uses classifier ensembling *plus LLM-based synthetic rephrasing* to reconstitute the ~90% of tokens that a DCLM/FineWeb-Edu-style aggressive filter would discard, blending "filter hard" and "synthesize more" into one pipeline — a direct descendant of both the DCLM classifier-filtering result and the Phi/"Textbooks" synthetic-data result. This is consistent with a broader 2025–2026 trend (seen across many non-headline papers turned up in this search, e.g. RefineX, arXiv:2507.03253) of using LLMs themselves as data-refinement tools rather than pure filters.

**Live controversies:**
1. **Copyright / scraping legality.** In June 2026, Digital Content Next sent Common Crawl (the raw-data backbone behind C4, RefinedWeb, Dolma, FineWeb, and DCLM) a cease-and-desist on behalf of AP, the New York Times, NBC, Bloomberg, NPR, and Fox, demanding it stop scraping and delete already-archived paywalled/member content. This sits alongside the ongoing NYT v. OpenAI suit (which specifically alleges Common Crawl made up ~60% of GPT-3's training mix) and Anthropic's ~$1.5B copyright class-action settlement (Sept 2025). If Common Crawl access is curtailed, essentially the entire open-data pipeline surveyed in §2.1 loses its raw material — a live, unresolved risk to the field's reproducibility model. (Sourced via WebSearch, not arXiv; these are legal/news events.)
2. **Model collapse from synthetic data.** A live empirical dispute: some results show even ~1% synthetic contamination degrades small models over generations (a "model collapse" effect), while other groups (and the production success of Phi-style synthetic-heavy training) argue collapse is avoidable as long as a sufficient fraction (informally cited around ≥5%) of real/verified data is mixed in each generation. Given Nemotron-CC alone now injects 1.9T synthetic tokens into a 6.3T-token corpus, this is not a hypothetical concern for current frontier pretraining.
3. **Dedup non-monotonicity.** FineWeb's finding that *more* deduplication (global, cross-snapshot MinHash) can *hurt* downstream performance versus lighter per-snapshot dedup complicates the simple "Lee et al. 2021: dedup is strictly good" takeaway — the current understanding is that dedup aggressiveness is itself a tunable hyperparameter with a non-monotonic effect, not a free win to maximize.

---

## 4. Standard textbooks / course notes

1. **Stanford CS336, "Language Modeling from Scratch"** (Percy Liang, Tatsunori Hashimoto et al.) — the explicit inspiration for this survey. Lecture 1 (Overview & Tokenization) walks through character/byte/BPE tokenization tradeoffs and has students implement BPE from scratch as Assignment 1; later lectures in the same course cover the data-curation stack (filtering, dedup, mixing). Course site: `cs336.stanford.edu`; lecture recordings publicly on YouTube.
2. **Karpathy, "Let's build the GPT Tokenizer"** (video + accompanying `minbpe` GitHub repo) — the most widely used from-scratch, code-first walkthrough of exactly how GPT-2/GPT-4-style byte-level BPE is trained and applied, including the regex-based pre-tokenization step and the `SolidGoldMagikarp`-style pathologies. Frequently paired with CS336 as a companion resource (e.g., referenced directly in CS336 student notes found during this search).
3. (Secondary/reference) **Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft**, Ch. 2 ("Regular Expressions, Tokenization, Edit Distance") — the standard NLP-textbook treatment of BPE/WordPiece tokenization, useful as a slower-paced supplement to CS336's lecture pace.

---

## 5. Concrete numbers for quiz questions

1. **GPT-2's byte-level BPE vocabulary is exactly 50,257 tokens** = 256 base byte tokens (one per possible byte value, guaranteeing zero `<unk>`) + 50,000 learned BPE merges + 1 special `<|endoftext|>` token. (This odd, non-power-of-2 number is also a commonly cited example of why later implementations pad the embedding matrix to 50,304 — the nearest multiple of 64 — for GPU tiling efficiency.)
2. **Tokenizer vocab growth across GPT generations:** GPT-2 → 50,257 (`gpt2`); GPT-3.5/GPT-4 → 100,256/100,277-ish (`cl100k_base`, commonly cited as ~100k); GPT-4o → **200,019** tokens (`o200k_base`) — roughly a 4x vocabulary expansion from GPT-2 to GPT-4o, driven mostly by better multilingual and code coverage.
3. **Compression ratio ("bytes/tokens" or equivalently chars/token):** OpenAI's own rule of thumb for English is roughly **4 characters ≈ 1 token** (≈0.75 words/token). Empirically, moving from the GPT-2 tokenizer to the GPT-4 (`cl100k_base`) tokenizer roughly **halves** the token count for the same English text (a commonly cited example: ~300 GPT-2 tokens vs. ~188 GPT-4 tokens for the same passage) — the compression-ratio improvement is itself a major, under-appreciated source of GPT-4's effective context-window/cost advantage over GPT-2/GPT-3.
4. **Tokenizer unfairness magnitude:** Petrov et al. (2023, arXiv:2305.15425) found the *same* sentence, translated into different languages, can require up to **15x more tokens** under a shared multilingual tokenizer — directly inflating API cost and shrinking effective context length for those languages.
5. **FineWeb scale:** 15 trillion tokens total, extracted from 96 Common Crawl snapshots; the **FineWeb-Edu** educational-quality-filtered subset is 1.3 trillion tokens. DCLM-Baseline (2024) retains only the **top ~10%** of documents (by fastText classifier score, threshold ≈0.018112 in the released code) from its raw candidate pool, yielding ~3.8T tokens across ~3B documents.
6. **Dedup can remove the majority of a naively-processed crawl:** FineWeb found that globally deduplicating the *oldest* CC snapshots against all 96 snapshots combined removed as much as **~90%** of that snapshot's post-filter data — and, counterintuitively, this over-aggressive global dedup *hurt* downstream model quality relative to lighter per-snapshot MinHash dedup (5-grams, 75% Jaccard similarity threshold).
7. **Data-constrained scaling:** Muennighoff et al. (2023, arXiv:2305.16264) found that repeating a fixed unique-token pool for **up to ~4 epochs** yields losses statistically indistinguishable from training on that many fresh unique tokens; the marginal value of each additional epoch beyond that decays roughly geometrically toward zero.
8. **Memorization from duplicates:** Lee et al. (2021, arXiv:2107.06499) found that **over 1%** of unprompted GPT-2-scale model output, prior to training-data deduplication, was verbatim-copied text from the training set — and deduplication reduced verbatim regurgitation roughly **10x**.

---

## Appendix: full verification ledger

`[API-VERIFIED]` = confirmed via a live `ti:"exact title"` call to `export.arxiv.org/api/query` during this session (either directly, or via a background retry job that eventually got through the rate limit).
`[PAGE-VERIFIED]` = the `export.arxiv.org/api/query` endpoint was persistently rate-limiting the shared sandbox IP (see note at top); confirmed instead by WebFetching the paper's own `arxiv.org/abs/<id>` page directly (same arxiv.org domain, different endpoint) and cross-checking the returned ID against an independent WebSearch hit showing the same `arxiv.org/abs/` URL.
NOT FOUND ON ARXIV = queried, zero results, not fabricated; genuinely lives elsewhere (blog, non-arXiv venue).

| # | Title | arXiv ID | Status |
|---|---|---|---|
| 1 | Neural Machine Translation of Rare Words with Subword Units | 1508.07909 | API-VERIFIED |
| 2 | Japanese and Korean Voice Search (WordPiece) | — (ICASSP 2012, not on arXiv) | Not on arXiv |
| 3 | Google's Neural Machine Translation System | 1609.08144 | API-VERIFIED |
| 4 | SentencePiece | 1808.06226 | API-VERIFIED |
| 5 | Subword Regularization | 1804.10959 | API-VERIFIED |
| 6 | Language Models are Unsupervised Multitask Learners (GPT-2) | — | NOT FOUND ON ARXIV (OpenAI report only) |
| 7 | ByT5 | 2105.13626 | API-VERIFIED |
| 8 | CANINE | 2103.06874 | API-VERIFIED |
| 9 | MEGABYTE | 2305.07185 | API-VERIFIED |
| 10 | MambaByte | 2401.13660 | API-VERIFIED |
| 11 | Byte Latent Transformer (BLT) | 2412.09871 | API-VERIFIED |
| 12 | Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net) | 2507.07955 | PAGE-VERIFIED |
| 13 | Fast Byte Latent Transformer | 2605.08044 | API-VERIFIED |
| 14 | Fishing for Magikarp | 2405.05417 | PAGE-VERIFIED |
| 15 | Glitch Tokens in LLMs: Categorization Taxonomy and Effective Detection | 2404.09894 | PAGE-VERIFIED |
| 16 | Investigating the Limitations of Transformers with Simple Arithmetic Tasks | 2102.13019 | PAGE-VERIFIED |
| 17 | Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks | 2305.14201 | WebSearch-cross-verified (title/authors/ID match direct arxiv.org listing) |
| 18 | Language Model Tokenizers Introduce Unfairness Between Languages | 2305.15425 | PAGE-VERIFIED |
| 19 | How Good is Your Tokenizer? | 2012.15613 | API-VERIFIED |
| 20 | Getting the most out of your tokenizer | 2402.01035 | API-VERIFIED |
| 21 | CCNet | 1911.00359 | API-VERIFIED |
| 22 | C4 / Exploring the Limits of Transfer Learning (T5) | 1910.10683 | API-VERIFIED |
| 23 | The Pile | 2101.00027 | API-VERIFIED |
| 24 | RefinedWeb | 2306.01116 | API-VERIFIED |
| 25 | Dolma | 2402.00159 | API-VERIFIED |
| 26 | FineWeb / FineWeb-Edu | 2406.17557 | API-VERIFIED |
| 27 | DataComp-LM (DCLM) | 2406.11794 | API-VERIFIED |
| 28 | Nemotron-CC | 2412.02595 | PAGE-VERIFIED |
| 29 | GPT-3 (Language Models are Few-Shot Learners) | 2005.14165 | API-VERIFIED |
| 30 | Gopher | 2112.11446 | API-VERIFIED |
| 31 | Quality at a Glance | 2103.12028 | PAGE-VERIFIED |
| 32 | Deduplicating Training Data Makes Language Models Better | 2107.06499 | API-VERIFIED |
| 33 | Deduplicating Training Data Mitigates Privacy Risks | 2202.06539 | API-VERIFIED |
| 34 | DoReMi | 2305.10429 | API-VERIFIED |
| 35 | Textbooks Are All You Need (phi-1) | 2306.11644 | PAGE-VERIFIED |
| 36 | Scaling Data-Constrained Language Models | 2305.16264 | PAGE-VERIFIED |

Non-arXiv sources cited (blog posts / news / legal, explicitly not claimed as arXiv papers): SolidGoldMagikarp original LessWrong post (Rumbelow & Watkins, Feb 2023); Broder's original MinHash paper (1997, pre-arXiv venue); Beren Millidge's "Integer tokenization is insane" blog series; Common Crawl cease-and-desist news coverage (2026); Anthropic copyright settlement news coverage (2025); OpenAI `tiktoken` GitHub repo (no paper).
