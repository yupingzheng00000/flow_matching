# LUT Refactor and Fix Plan

**Overall Progress:** `100%`

## Tasks:

- 🟩 **Step 1: Centralize LUT distance at path level**
  - 🟩 Use path-level `metric_name`/`lp_order` in `_build_lut_distance_table`
  - 🟩 Apply normalized distance only for vector embeddings (`emb_dim > 1`)
  - 🟩 Preserve cosine handling and caching semantics
- 🟩 **Step 2: Add interface parity to LearnableParametricLUT**
  - 🟩 Expose `num_channels`, `vocab_size`, `emb_dim`, and `embed_range`
  - 🟩 Add renorm/bounded-scale options with per-channel base norms
  - 🟩 Ensure `weight` property returns the forward tensor
- 🟩 **Step 3: Deduplicate small utilities**
  - 🟩 Remove duplicate softplus inverse helper
  - 🟩 Share linear baseline helper across LUT implementations
- 🟩 **Step 4: Keep channel mapping as-is**
  - 🟩 Reuse existing block mapping via `num_channels` property
- 🟩 **Step 5: Align tests and usages**
  - 🟩 Update LUT tests to accept `(C, V, D)` shapes and new buffers
  - 🟩 Verify diagnostics and training paths remain compatible
- 🟩 **Step 6: Save/Load behavior (keep simple)**
  - 🟩 Leave checkpoint load/save logic unchanged (strict load maintained)
