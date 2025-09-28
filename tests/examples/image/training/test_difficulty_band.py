import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLES_ROOT = ROOT / "examples" / "image"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from examples.image.training.train_loop import (
    DifficultyBandState,
    _compute_gap_quantiles,
    _map_gaps_to_logbeta_band,
)
from flow_matching.path.mixture import MetricInducedGibbsProbPath


def test_map_gaps_to_logbeta_band_basic():
    q10 = 0.1
    q90 = 0.2
    ratio_min = 0.5
    ratio_max = 0.05

    result = _map_gaps_to_logbeta_band(q10, q90, ratio_min, ratio_max)
    assert result is not None
    ell_min, ell_max = result

    beta_min = math.log(1.0 / ratio_min) / q90
    beta_max = math.log(1.0 / ratio_max) / q10
    expected_min = math.log(beta_min)
    expected_max = math.log(beta_max)

    assert math.isclose(ell_min, expected_min, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(ell_max, expected_max, rel_tol=1e-6, abs_tol=1e-6)


@torch.no_grad()
def test_difficulty_band_refresh_updates_state():
    path = MetricInducedGibbsProbPath(vocab_size=16, emb_dim=1)
    tokens = torch.arange(0, 16, dtype=torch.long).view(4, 4)

    state = DifficultyBandState(lmin=0.5, lmax=2.2)
    refreshed = state.refresh(
        tokens=tokens,
        path=path,
        step=5,
        sample_fraction=1.0,
        max_positions=64,
        ratio_min=0.5,
        ratio_max=0.05,
    )

    assert refreshed is True
    assert state.has_stats
    assert state.last_refresh_step == 5

    distances = path.distances_from_tokens(tokens.view(-1, 1)).squeeze(1)
    q10, q90 = _compute_gap_quantiles(distances)
    assert math.isclose(state.q10 or 0.0, q10, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(state.q90 or 0.0, q90, rel_tol=1e-6, abs_tol=1e-6)

    expected_band = _map_gaps_to_logbeta_band(q10, q90, 0.5, 0.05)
    assert expected_band is not None
    ell_min, ell_max = expected_band
    assert math.isclose(state.lmin, ell_min, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(state.lmax, ell_max, rel_tol=1e-6, abs_tol=1e-6)
    assert state.interval() > 0.0
