# KO-DFM / NFDM Feature Spec (CIFAR-10, discrete pixels)

> Owner: YOU (research), Implementer: coding engineer
> Scope: **discrete Flow Matching** with KO velocities; **metric-induced path** (Euclidean on pixel values) with a **learnable scheduler $\beta_t$**; **UNet** backbone; **factorized x-pred** training; **single-token flips** in CTMC.

---

## 0) High-level goals

* Keep **source prior uniform i.i.d.** over tokens; work with **conditional probability paths** per token.
* Pretrain with fixed path/schedule; then **learn a monotone scheduler** $\beta_t$ while keeping the metric fixed (Euclidean).
* For sampling, support:

  1. **Mixture path solver** (library provided) for baselines; and
  2. **KO analytic conditional velocity** for metric-induced path (simple Euler CTMC stepper).
* Do **not** parameterize rate kernels; derive them from the path (KO).

---

## 1) Repo touchpoints (files to add/modify)

### 1.1 Models

* **`models/unet_discrete.py` (new or adapt existing)**

  * **Input stem**: `nn.Embedding(K=256, C_in=96)`. Given `x_t: LongTensor [B, 3, H, W]`, produce `[B, 3*C_in, H, W]` by lookup and channel-stack.
  * **Time/schedule conditioning**: existing `t` embedding (sinusoidal/MLP) injected into UNet as usual.
  * **Head**: final `1×1 Conv` to `3*256` channels; reshape to logits `[B, 3, H, W, 256]` (or `[B, d, K]`).
  * **No one-hot tensors** go through the UNet; only the embedded feature map.

### 1.2 Paths & schedulers

* **`paths/mixture.py` (new)**

  * Class: `MetricInducedGibbsProbPath(ProbPath)`
  * Purpose: **sampling $X_t$** from **metric-induced Gibbs conditional**
    $ p_t(x_i\mid x^{(1)}_i) \propto \exp\{-\beta_t\, d(E[x_i], E[x^{(1)}_i])\}$
  * Distance: start with **Euclidean** over a deterministic embedding `E[k]=(2/255)x - 1` (optionally width>1 by tiling).
  * API:

    * `sample(x_0: Long[B,S], x_1: Long[B,S], t: Float[B]) -> DiscretePathSample`
    * `beta(t) -> (beta_t, d_beta_t)` with stable clamp of `t`.
  * **Note**: this class is used for **training-time sampling** of `x_t`; it does **not** implement the mixture velocity.

* **`schedulers/poly_convex.py` (new/not for now)**

  * Class: `PolyConvexScheduler(n: float|Tensor = learnable, c: float=1.0)`
  * Implements monotone convex schedule on **logit-time**: $\beta(t) = c\,(t/(1-t))^n$.
  * Exposes `forward(t) -> beta_t`, `derivative(t) -> d_beta_t` (analytical).
  * Register as a drop-in for places expecting a `ConvexScheduler`.

* **(Optional)** `schedulers/monotone_rqs.py`

  * Class: `LogitTimeRQSScheduler(K=8)` (monotone rational-quadratic spline on `logit(t)`) with built-in positivity for $\dot\beta_t$.
  * Methods: `forward`, `derivative`, `init_linear`.

### 1.3 Training loop glue

* **`trainers/dfm_discrete.py` (modify/new)**

  * Sample `t ~ U[0,1]` per batch; **draw `x_t` from `MetricInducedGibbsProbPath.sample`** using current scheduler.
  * UNet forward: `logits = unet(x_t, t)`; **shape last-dim `K=256`**.
  * Loss: `MixturePathGeneralizedKL(path_mixture)` where `path_mixture` is **instantiated only to supply the scheduler** (it expects a `MixtureDiscreteProbPath`). For metric-induced experiments, $\kappa(t)$ should match our `beta` schedule mapping; keep this in one place (see §2.3 Contracts).
  * Logging: bits/dim, CE, and KL terms (from the loss) vs. `t`.

### 1.4 Inference

* **`samplers/ko_ctmc_metric.py` (new)**

  * Minimal first-order **Euler CTMC** stepper that, per site `i`, uses the **KO analytic conditional velocity** for metric-induced paths:
    $u_t^i(x, z\mid x_1) = p_t(x\mid x_1)\, \dot\beta_t\, [d(z,x_1) - d(x,x_1)]_+$
  * Loop: for each `i`, compute exit rate `lambda_i = sum_{x\neq z} u_t^i(x,z|x_1)`; flip token with prob `1 - exp(-h*lambda_i)`; if flips, sample new state from normalized `u_t^i` (single-flip constraint).
  * Also support a **divergence-free corrector** weight `gamma` (optional), no-op by default.

---

## 2) Interfaces & contracts (what to code to)

### 2.1 Shapes

* **Model output**: logits over K per site, **last dim = K**; either `[B, d, K]` or `[B, 3, H, W, K]`.
* **Tokens**: `x_t, x_1` are **integer** tensors, shapes `[B, d]` (flattened) or `[B, 3, H, W]`.
* **Path sample**: `DiscretePathSample(x_t, x_1, x_0, t)`.

### 2.2 UNet I/O

* **Input**: integer tokens → `nn.Embedding(256, 96)` → `[B, 288, 32, 32]` into UNet.
* **Output**: `1×1 Conv` to `768` → reshape to `[B, 3, 32, 32, 256]`.

### 2.3 Schedulers (one true source of time)

* Keep **one scheduler object** with `beta(t)` and `d_beta(t)`; both **training sampler** and **loss** must read from it to avoid mismatch.
* For **MixturePathGeneralizedKL**: supply a `MixtureDiscreteProbPath(scheduler=EquivalentConvex(beta↦kappa))` to meet the type and read `\kappa_t, \dot\kappa_t` from the same params.

---

## 3) Detailed tasks (engineering)

### T1 — UNet adapters (input & head)

* Add `nn.Embedding(256, 96)`; lookup per pixel/channel; stack into 288-ch input.
* Replace first conv with identity (or keep it; both fine).
* Final layer to `3*256` channels; provide utility `reshape_logits()` to make `[B, d, K]`.

### T2 — Metric-induced path sampler

* Implement `MetricInducedGibbsProbPath` with deterministic `E[k]=k/255` (float32).
* Distance via `torch.cdist` (Euclidean) for stability; optional `cosine` flag.
* Stable scheduler `beta(t) = c*(t/(1-t))^n` with clamps on `t` and analytic derivative.
* `sample()` draws per-site categories with `torch.multinomial` (replacement=True) from softmax(-beta\*d).

### T3 — Learnable scheduler(s)

* `PolyConvexScheduler`: parameter `n` is **learnable scalar** (positive via `softplus` + 1), constant `c>0` learnable if desired.
* (Optional) `LogitTimeRQSScheduler`: monotone spline on `logit(t)`; store knots in ascending order; compute analytic `d_beta`.

### T4 — Training loop glue

* New dataclass `TrainCfg` to toggle `path_kind in {mixture, metric}`, `scheduler_kind`, and `learn_scheduler`.
* If `learn_scheduler=True`, choose a **differentiable estimator** to backprop through `x_t` draw: **Gumbel-Softmax** (temperature anneal) or **enumeration** over a small K (for ablations).
* Loss remains `MixturePathGeneralizedKL` (x-pred), fed with logits, `x_1`, `x_t`, `t`.

### T5 — KO metric CTMC sampler (inference)

* Implement `step(x_t, x_1, t, h)` with the KO conditional velocity; supports `div_free=0.0` initially.
* Provide `sample(x_init, time_grid, step_size)` returning final tokens (and optional intermediates for FID tracking).

### T6 — Evaluation harness

* CIFAR-10 32×32, K=256 per channel.
* Metrics: FID-50k, Inception score; NFE sweep (e.g., 20/50/100).
* Baselines: mixture path + official solver; our metric-induced + KO stepper (same NFE/time grid).
* Ablations: fixed vs learned `n`; Euclidean vs cosine metric.

---

## 4) Acceptance criteria

* ✔ **Training** runs without shape errors; CE/DFM loss finite across `t∈(0,1)`.
* ✔ **Scheduler** parameters show gradient flow when `learn_scheduler=True` (check `.grad` nonzero; loss decreases vs. learned `n`).
* ✔ **Sampling**: mixture baseline reproduces library behavior; KO stepper generates valid PMFs each step (`probs.sum(-1)==1`, no negatives).
* ✔ **Qualitative**: images sharpen as `t→1`; fewer artifacts with larger $\beta_t$.
* ✔ **Quantitative**: metric-induced path + learned `n` matches or beats mixture baseline at the same NFE.

---

## 5) Risks & mitigations

* **Library coupling (loss expects mixture path type):** keep a tiny `MixtureDiscreteProbPath(scheduler)` solely to provide $\kappa_t, \dot\kappa_t$ to the loss, while sampling `x_t` from our metric path.
* **Gradient through `x_t`:** when learning the scheduler, use **Gumbel-Softmax** relaxation; anneal temperature; or freeze scheduler for first K epochs.
* **Numeric stability near `t≈1`:** clamp `t∈[ε,1-ε]`; cap `beta_t` in logs; compute `d_beta_t` analytically.

---

## 6) Minimal pseudocode (wire-up)

```python
# x1: Long [B, 3, H, W]; t: Float [B]
# 1) draw xt under metric-induced Gibbs path
xt = metric_path.sample(x_0=torch.empty_like(x1[...,0,0]),  # ignored
                        x_1=x1.view(B, -1), t=t).x_t.view(B, 3, H, W)

# 2) model forward
logits = unet(xt, t)       # [B, 3, H, W, 256]
logits_flat = logits.view(B, -1, 256)

# 3) loss (mixture DFM, just to read scheduler)
loss = dfm_loss(logits_flat, x1.view(B,-1), xt.view(B,-1), t)
loss.backward(); opt.step()
```

---

## 7) Stretch (later)

* Add **divergence-free corrector** in the KO stepper (tunable `γ`).
* Support **cosine** metric with a semantic table `E`.
* Learn a **residual metric** `d_ψ` under monotonic constraints; keep KO velocity via closed-form or a single Laplacian solve.

---

## 8) Glossary

* **x-pred head**: model outputs $p(x_1\mid x_t)$ logits per token.
* **Single-flip**: only one token changes at a time in the CTMC step.
* **KO**: kinetic-optimal construction that decouples probability path from velocity and gives valid rates.
