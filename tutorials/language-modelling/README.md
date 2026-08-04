# Tutorial: a language model, from scratch

You read the explainer. Now write it. Twelve numpy functions — the tokenizer,
the objective, the transformer block, the optimizer, the sampler, and the
resource accounting — with a test suite that knows the right answers.

Nothing here is a toy identity for its own sake. Every expected value is
either something you can derive on paper (a uniform model over V tokens has
perplexity exactly V) or a number the explainer quizzes you on.

## How to work

```bash
cd tutorials/language-modelling
python3 -m pytest -q          # red: 32 failing tests
```

Open `lm.py` and replace each `raise NotImplementedError`, re-running the
tests as you go. Suggested order (easiest → hardest):

1. `softmax` — subtract the max first, or you will meet your first NaN
2. `rmsnorm` — note what is missing versus LayerNorm
3. `cross_entropy`, `perplexity` — via log-sum-exp
4. `swiglu` — three matrices, one gate
5. `rope` — the rotation whose dot product only sees relative distance
6. `attention` — mask *before* the softmax
7. `adamw_step` — decoupled decay is the whole difference from Adam
8. `train_bpe`, `bpe_encode` — merge the most frequent pair, repeat
9. `sample_next` — temperature, then top-k, then top-p, then renormalize
10. `transformer_flops`, `chinchilla_optimal` — the accounting behind C ≈ 6ND

Check your work against the reference at any point:

```bash
LM_SOLUTION=1 python3 -m pytest -q     # green: 32 passing
```

## The two integration tests

These are the ones worth reaching for. They only pass when several pieces are
correct together:

- **`test_end_to_end_induction_head_copies_the_right_token`** builds, by hand,
  the attention head that underlies in-context learning: keys record "what came
  before me", so a query matching an earlier symbol retrieves whatever followed
  it. Given `A B C A` your head must predict `B`. No training involved — the
  mechanism is pure attention, and once you have written `attention` and
  `softmax` correctly it simply works.

- **`test_end_to_end_a_trained_bpe_shrinks_a_real_corpus_and_lowers_perplexity`**
  trains your BPE on real text and shows the compression paying for itself:
  fewer, longer tokens beat raw bytes in nats per byte even though the
  vocabulary — and therefore the per-token entropy ceiling — got larger. That
  trade is the entire reason tokenizers exist.

## Things the tests will teach you the hard way

- `softmax` on logits of 1000 returns NaN unless you shift by the max.
- Masking attention *after* the softmax silently breaks normalization; the
  causal test catches it.
- AdamW's first step is `lr·sign(g)` no matter the gradient scale — but only
  while `|g| ≫ eps`. There is a test for each half of that sentence.
- BPE training stops early when no adjacent pair repeats twice, so a small
  corpus cannot fill a large vocabulary. Real tokenizers need real data.
- `top_p` must keep the token that *crosses* the threshold, not stop before it.
