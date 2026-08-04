#!/usr/bin/env python3
"""Verify cited papers against the arXiv API, with backoff + an on-disk cache.

Safe to re-run: anything already resolved is skipped, so repeated runs only
retry the gaps. Writes survey/verified.json and prints a table.
"""
import json, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

API = "http://export.arxiv.org/api/query?"
NS = {"a": "http://www.w3.org/2005/Atom"}
HERE = Path(__file__).resolve().parent
CACHE = HERE / "verified.json"
DELAY = 11.0         # generous: arXiv throttles bursts hard
COOLDOWN = 120       # let an existing rate-limit window expire before starting

TITLES = [
 "Attention Is All You Need",
 "Language Models are Few-Shot Learners",
 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
 "Root Mean Square Layer Normalization",
 "On Layer Normalization in the Transformer Architecture",
 "RoFormer: Enhanced Transformer with Rotary Position Embedding",
 "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
 "GLU Variants Improve Transformer",
 "Fast Transformer Decoding: One Write-Head is All You Need",
 "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
 "YaRN: Efficient Context Window Extension of Large Language Models",
 "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
 "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
 "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models",
 "DeepSeek-V3 Technical Report",
 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
 "Neural Machine Translation of Rare Words with Subword Units",
 "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing",
 "Byte Latent Transformer: Patches Scale Better Than Tokens",
 "Deduplicating Training Data Makes Language Models Better",
 "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale",
 "DataComp-LM: In search of the next generation of training sets for language models",
 "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
 "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining",
 "Scaling Data-Constrained Language Models",
 "Adam: A Method for Stochastic Optimization",
 "Decoupled Weight Decay Regularization",
 "Scaling Laws for Neural Language Models",
 "Training Compute-Optimal Large Language Models",
 "Chinchilla Scaling: A replication attempt",
 "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws",
 "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer",
 "Muon is Scalable for LLM Training",
 "An Empirical Model of Large-Batch Training",
 "Mixed Precision Training",
 "Are Emergent Abilities of Large Language Models a Mirage?",
 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
 "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
 "Training Deep Nets with Sublinear Memory Cost",
 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
 "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
 "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism",
 "Ring Attention with Blockwise Transformers for Near-Infinite Context",
 "Efficient Memory Management for Large Language Model Serving with PagedAttention",
 "Fast Inference from Transformers via Speculative Decoding",
 "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers",
 "Training language models to follow instructions with human feedback",
 "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
 "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
 "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
 "Proximal Policy Optimization Algorithms",
 "s1: Simple test-time scaling",
 "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?",
]

db = json.loads(CACHE.read_text()) if CACHE.exists() else {}
import sys
if "--cooldown" in sys.argv:
    print(f"cooling down {COOLDOWN}s before first request", flush=True)
    time.sleep(COOLDOWN)

def fetch(title, tries=8):
    url = API + urllib.parse.urlencode(
        {"search_query": 'ti:"%s"' % title, "max_results": 3})
    wait = DELAY
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return ET.fromstring(r.read())
        except Exception as e:
            if attempt == tries - 1:
                raise
            wait *= 1.8                       # exponential backoff on 429/timeouts
            print(f"   retry {attempt+1} ({e}) in {wait:.0f}s", flush=True)
            time.sleep(wait)

def parse(root, title):
    best = None
    for e in root.findall("a:entry", NS):
        t = " ".join(e.find("a:title", NS).text.split())
        rec = {"title": t,
               "id": e.find("a:id", NS).text.rsplit("/", 1)[-1],
               "published": e.find("a:published", NS).text[:10],
               "first_author": e.find("a:author/a:name", NS).text}
        if t.lower() == title.lower():
            rec["match"] = "EXACT"; return rec
        if best is None:
            best = dict(rec, match="FUZZY")
    return best

todo = [t for t in TITLES if t not in db or not db[t]]
print(f"{len(db)} cached, {len(todo)} to fetch\n", flush=True)
for t in todo:
    try:
        db[t] = parse(fetch(t), t)
    except Exception as e:
        print("FAILED", t, e, flush=True)
        continue
    CACHE.write_text(json.dumps(db, indent=1))
    r = db[t]
    print("%-5s | %-13s | %-10s | %s" %
          (r["match"] if r else "NONE", r["id"] if r else "-",
           r["published"] if r else "-", (r["title"] if r else t)[:72]), flush=True)
    time.sleep(DELAY)

ok = sum(1 for t in TITLES if db.get(t))
print(f"\nresolved {ok}/{len(TITLES)}")
miss = [t for t in TITLES if not db.get(t)]
if miss:
    print("STILL MISSING:"); [print("  -", m) for m in miss]
