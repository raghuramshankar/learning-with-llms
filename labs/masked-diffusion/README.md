# Lab: masked diffusion, from scratch

You read the explainer. Now write it. Six small numpy functions — the forward
masking process, the reveal-probability sampler, the NELBO weight, the weighted
MLM loss, and a DDIM step — with a test suite that knows the right answers,
including statistical checks that your sampler actually preserves the marginals.

## How to work

```bash
cd labs/masked-diffusion
python3 -m pytest -q          # red: 12 failing tests
```

Open `masked_diffusion.py` and replace each `raise NotImplementedError`,
re-running the tests as you go. Suggested order (easiest → hardest):

1. `reveal_prob` — one formula from Maths II
2. `nelbo_weight` — one line after you simplify by hand
3. `forward_mask` — one vectorized `np.where`
4. `weighted_mlm_loss` — boolean indexing
5. `ddim_step` — two lines from Maths I
6. `reverse_step` — the real one: carry-over + reveal + sample from the model

When the suite is green, the integration test has verified that your reverse
chain, driven by an oracle model, reconstructs a sentence exactly from
all-masked — i.e. you have implemented a working masked-diffusion sampler.

## Rules of engagement

- The tests are the spec: read a failing test before asking what a function
  should do.
- `solutions/masked_diffusion.py` exists. Peeking at **one function** after a
  real attempt is studying; reading the file is not.
- Everything needed is in the explainer:
  `docs/2026-08-01-diffusion-language-models.html` (Maths I and Maths II).
