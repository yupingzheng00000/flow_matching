# Research Notes
- Runtime error occurs during metric-induced training when computing schedule KL value: `(kl_tensor * logbeta_weights)` fails because `kl_tensor` has shape `(batch, num_tokens)` while `logbeta_weights` has shape `(batch,)`.
- Importance weights from `sample_t_uniform_logbeta` correspond to per-sample factors and should broadcast across token dimension in KL calculation.
- Need to reshape or reduce KL tensor to avoid shape mismatch.
