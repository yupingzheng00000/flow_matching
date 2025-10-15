# Implementation Detail Doc — 1-D Per-Channel LUT for Metric-Induced Path (CIFAR-10)

## Scope (the “simplest setting”)

Goal: replace the baseline scalar embedding `emb(x)=2x/255−1` used **inside the probability path** with a **learned 1-D per-channel lookup table (LUT)**, with:

* **No regularizers** (first test),
* **Linear initialization** (start equal to the baseline mapping),
* **Factorized per-channel path** (256-way categorical per channel, not joint RGB),
* Clear mechanics for **computing distances during sampling**.

This document only touches the **probability path** (p_t)—architecture (U-Net) can remain as in the Dhariwal–Nichol U-Net variant commonly used for diffusion, and, if you later follow DFM’s CIFAR-10 wiring (token embedding input, per-channel categorical head), that is orthogonal to this path definition. ([Proceedings of Machine Learning Research][1])

---

## Notation & setup

* Pixel at site (i) has channels (x_i=(R_i,G_i,B_i)), each (x_{ic}\in{0,\dots,255}).
* For each channel (c\in{R,G,B}) we learn a **scalar** embedding table
  (E_c:{0,\dots,255}\rightarrow\mathbb{R}) (shape (256\times 1)).
  Collectively, (E) has shape ((3,256,1)).
* The **metric-induced conditional path** (factorized per channel) is
  [
  p_t(x_{ic}=v\mid x_{1,ic}) ;\propto; \exp\big(-\beta(t),d_c(v, x_{1,ic})\big),\quad v\in{0,\dots,255},
  ]
  where (d_c(v,x)=\big|,E_c[v]-E_c[x],\big|) (one-dimensional absolute difference).
  (In one dimension, all (\ell_p) norms reduce to the absolute value; if you later want a “sharper” distance with 1-D embeddings, you would explicitly use (|\cdot|^p) as a design choice, not a norm.)
* **Scheduler**: use your chosen (\beta(t)) (e.g., power-law). DFM emphasizes that **general probability paths** and schedules are valid; the velocity model is trained against the path you choose. ([arXiv][2])

Why token-wise (per-channel) is standard: discrete diffusion/flow papers shape transitions/paths at the **symbol level** (e.g., 256-way image tokens) via structured distances/embeddings, rather than a joint RGB categorical over (256^3). ([NeurIPS Papers][3])

---

## Component 1 — The LUTs (what they are)

* **Definition**: three tables (E_R,E_G,E_B), each length 256 with 1 scalar per token (total parameters (3\times256)).
* **Initialization (linear)**:
  [
  E_c[v] \leftarrow 2\frac{v}{255}-1,\quad v=0,\dots,255,
  ]
  reproduced per channel. This **exactly matches** the baseline scalar embedding at step 0. (You can add a mask entry later if you adopt mask-based training; that would be a 257th entry as done in DFM for architecture—but the **path** here is unchanged.) ([NeurIPS 会议录][4])
* **Training**: treat (E) as learnable parameters updated by your main optimizer (same LR as the model is fine for a first run).

---

## Component 2 — Distance used by the path (per channel)

For a given pixel/channel, define the **scalar** distance:
[
d_c(v,x);=;\big|;E_c[v]-E_c[x];\big|.
]

* Because (E_c[\cdot]\in\mathbb{R}), (\ell_p) collapses to (|\cdot|). If you later need a distinct “(p)” effect without increasing dimension, you must raise the absolute to a power (design choice), or increase the embedding dimension (D>1) and use a true (\ell_p) in (\mathbb{R}^D).
* Keeping it **1-D** is the simplest, least invasive variant and aligns with DFM’s spirit of flexible probability paths without complicating normalization. ([NeurIPS Papers][3])

---

## Component 3 — Computing logits & sampling during training

You need to **sample (x_t)** from the path to feed the network during training (teacher forcing / conditional sampling). For each channel (c), at time (t):

### Inputs

* (x_{1,ic}) at each spatial site (i) (the “clean” token for channel (c)),
* Time (t) (scalar or per-batch),
* LUT (E_c) (length-256 vector of scalars),
* Scheduler (\beta(t)).

### Steps (vectorized over batch/spatial)

1. **Gather center embedding per site**
   (e_{\text{ctr}}(i) \leftarrow E_c[x_{1,ic}]) producing a tensor of shape ((B,H,W,1)) for channel (c).
2. **Prepare candidate embeddings**
   (e_{\text{cand}}(v) \leftarrow E_c[v]) for all (v=0,\dots,255), arranged as ((1,1,1,256,1)) so it broadcasts against all sites.
3. **Compute distances to all candidates**
   [
   d(i,v);=;\big|,e_{\text{cand}}(v)-e_{\text{ctr}}(i),\big|
   \quad\Rightarrow\quad d\in\mathbb{R}^{B\times H\times W\times 256}.
   ]
4. **Convert to logits with the scheduler**
   [
   \text{logits}(i,v) \leftarrow -\beta(t)\cdot d(i,v).
   ]
   (Broadcast (\beta(t)) to ((B,1,1,1)) as needed.)
5. **Normalize & sample**
   Apply softmax over the **256 candidates** (the channel’s vocabulary) to get (p_t(\cdot\mid x_{1,ic})), then draw one sample per site.

   * If you use Gumbel noise for hard sampling, inject it **before** argmax (Gumbel-Max trick).
   * This mirrors token-wise categorical sampling used in discrete diffusion/flow (e.g., structured transition matrices in D3PM; mask-based discrete models). ([NeurIPS Papers][3])

Repeat the same for (c\in{R,G,B}) and assemble (x_t=(x_{tR},x_{tG},x_{tB})).

**Complexity**: per channel, you compute 256 distances per pixel. This is the standard **256-way categorical** cost used by discrete models and is tractable on CIFAR-10. ([NeurIPS Papers][3])

---

## What **doesn’t** change

* **Factorization**: Path remains **per-channel**, not joint RGB; you are not normalizing over (256^3). This is consistent with discrete diffusion practice where transitions/paths operate at the token level with structured geometry. ([NeurIPS Papers][3])
* **Network**: Your U-Net body, loss, and training loop stay the same. If you later adopt DFM’s CIFAR-10 U-Net wiring (replace first layer with an input token-embedding and widen the head to output per-channel categoricals), that is **architectural** and compatible with this path; the two ideas are decoupled. ([NeurIPS 会议录][4])
* **Objectives**: Use your existing discrete FM/DFM training objective. DFM’s framework explicitly allows **general probability paths** like this metric-induced Gibbs path. ([arXiv][2])

---

## Sanity checklist for the first run (no regularizers)

* **Initialization** reproduces the baseline: check that (E_c[v]=2v/255-1) at step 0 and that distance histograms match the original path.
* **Scheduler range**: verify (\beta(t)) is numerically stable near (t\in(0,1)) (clamp if needed).
* **Sampling entropy**: measure the entropy of (p_t(\cdot\mid x_{1c})) at a few (t) values; if it collapses (too peaky) or explodes (too flat), adjust (\beta(t)) scale before considering any regularizers. (DFM highlights sensitivity to path/scheduler choices—monitoring the path’s effective support is good practice.) ([arXiv][2])

---

## References / context

* **Discrete Flow Matching (DFM)** — general probability paths; CIFAR-10 architectural edits (token embedding input; per-channel categorical head). ([arXiv][2])
* **D3PM** — structured discrete diffusion with token-wise transition matrices shaped by embedding-space neighborhoods and absorbing states (conceptual precedent for embedding-shaped token distances). ([NeurIPS Papers][3])
* **U-Net baseline** (Dhariwal & Nichol 2021) — the common diffusion U-Net backbone referenced by DFM for CIFAR-10 setups. ([Proceedings of Machine Learning Research][1])

---

### TL;DR

* **What you add**: three 1-D LUTs (E_R,E_G,E_B) (linear init).
* **What you change**: the **distance** inside the Gibbs path becomes (|E_c[v]-E_c[x_{1c}]|) per channel.
* **How you sample**: compute 256 distances → logits (-\beta(t)\cdot d) → softmax (per channel) → sample.
* **Everything else stays** the same. This is the smallest, cleanest way to learn a better token geometry for the path—fully aligned with DFM/D3PM’s discrete-token view. ([arXiv][2])

[1]: https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf?utm_source=chatgpt.com "Improved Denoising Diffusion Probabilistic Models"
[2]: https://arxiv.org/abs/2407.15595?utm_source=chatgpt.com "[2407.15595] Discrete Flow Matching"
[3]: https://papers.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf?utm_source=chatgpt.com "Structured Denoising Diffusion Models in Discrete State- ..."
[4]: https://proceedings.neurips.cc/paper_files/paper/2024/file/f0d629a734b56a642701bba7bc8bb3ed-Paper-Conference.pdf?utm_source=chatgpt.com "Discrete Flow Matching"
