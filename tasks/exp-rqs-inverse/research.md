# Research: Exp RQS inverse implementation

- `_ExpRQS1D.forward` constructs monotone RQS with linear tails; parameters derived from `theta_w`, `theta_h`, `theta_d`.
- Forward builds `xk`, `yk`, `delta` exactly as specified by NSF. Inverse must reconstruct identical arrays to ensure consistency.
- For input in central region, we need to identify bin via `torch.bucketize` over `xk[1:]` or `yk[1:]` depending on value to invert.
- Quadratic coefficients from NSF for solving normalized coordinate ξ given normalized output `z = (slope*ξ^2 + d0*ξ(1-ξ)) / (slope + (d0 + d1 - 2*slope)*ξ(1-ξ))` are:
  - `a = slope - d0 + z * (d0 + d1 - 2 * slope)`
  - `b = d0 - z * (d0 + d1 - 2 * slope)`
  - `c = -z * slope`
- Tails remain linear with slope 1 ⇒ inverse is identity there; derivative = 1.
- To avoid numerical issues, we need eps for denominators and clamp discriminant >=0. Use root `(-b + sqrt(disc)) / (2*a)` (monotone case) and clamp to [0,1].
- After solving, map `s = x0 + w * xi`. Then compute derivative using forward at `s` to reuse existing code.
- Tests should draw random inputs within [-tail_bound, tail_bound], check inverse(forward(x)[0])[0] ≈ x, also test outside (tails) and gradient shape.
