import dataclasses

import numpy as np
import pytest
import torch

from byosm.series.augment import amplitude, augmented, censor, hidden, mix
from byosm.series.metrics import QUANTILES
from byosm.series.models import Group
from byosm.series.patch_net import MAX_HORIZON, SMALL, TARGET_CAP, PatchNet, Size, batch, window
from byosm.series.recipe import RECIPES, TRAINING_QUANTILES, V1, V2
from byosm.series.synthetic import MIN_LENGTH, Synthetic

TINY = Size(context=64, patch=16, width=16, layers=1, heads=2)


def _wave(n=600, seed=0):
    t = np.arange(n)
    return 10 + np.sin(2 * np.pi * t / 24) + np.random.default_rng(seed).normal(0, 0.1, n)


def test_the_v2_recipe_trains_21_quantiles_and_serves_the_9_standard_levels_as_a_subset():
    assert len(TRAINING_QUANTILES) == 21 and TRAINING_QUANTILES[0] == 0.01 and TRAINING_QUANTILES[-1] == 0.99
    assert [V2.quantiles[i] for i in V2.served] == list(QUANTILES) and V1.served == list(range(len(QUANTILES)))
    net = PatchNet(SMALL, len(V2.quantiles))
    assert net(torch.zeros(2, SMALL.context), torch.ones(2, SMALL.context)).shape == (2, 21, MAX_HORIZON)
    assert set(RECIPES) == {"v1", "v2"} and V1.weights("base") == "base" and V2.weights("base") == "base-v2"


def test_arcsinh_scaling_is_inverted_on_the_way_out_and_keeps_the_order():
    x = np.linspace(-200, 200, 101)
    assert np.allclose(V2.unscale(V2.scale(x)), x) and np.all(np.diff(V2.scale(x)) > 0)
    assert V1.scale(x) is x and V1.unscale(x) is x


def test_the_v1_recipe_draws_the_same_windows_and_targets_as_before():
    rng = np.random.default_rng(0)
    sizes = zip(rng.integers(100, 900, 12), rng.integers(8, 61, 12), strict=True)
    chosen = [(rng.normal(5, 2, int(n)), int(h)) for n, h in sizes]
    expected, draw = [[], [], [], []], np.random.default_rng(3)
    for h, horizon in chosen:
        end = int(draw.integers(SMALL.patch, len(h) - horizon + 1))
        v, m, mean, std = window(h[:end], SMALL.context)
        future = np.full(MAX_HORIZON, np.nan)
        future[:horizon] = np.clip((h[end : end + horizon] - mean) / std, -TARGET_CAP, TARGET_CAP)
        for part, value in zip(expected, (v, m, np.nan_to_num(future), ~np.isnan(future)), strict=True):
            part.append(value)
    got = batch(chosen, SMALL, np.random.default_rng(3), "cpu", V1)
    tensors = [torch.tensor(np.asarray(e), dtype=torch.float32) for e in expected]
    assert all(torch.equal(g, e) for g, e in zip(got, tensors, strict=True))


def test_patch_masking_hides_one_contiguous_span_of_whole_patches_at_the_named_rate():
    rng, past = np.random.default_rng(0), np.arange(32 * 20, dtype=float)
    shares = []
    for _ in range(400):
        gone = np.flatnonzero(np.isnan(hidden(past, 32, 0.15, rng)))
        if len(gone):
            assert gone[0] % 32 == 0 and len(gone) % 32 == 0 and np.all(np.diff(gone) == 1)
        shares.append(len(gone) / len(past))
    assert abs(np.mean(shares) - 0.15) < 0.02 and hidden(past, 32, 0.0, rng) is past
    short = np.full(32 * 20, np.nan)
    short[-32:] = 1.0
    assert all(np.isfinite(hidden(short, 32, 0.5, rng)).sum() == 32 for _ in range(100))


def test_amplitude_modulation_keeps_the_level_and_censoring_clips_one_side():
    rng, segment = np.random.default_rng(0), _wave(200)
    segment[150] = np.nan
    modulated = amplitude(segment, 140, rng)
    assert abs(np.nanmean(modulated[:140]) - np.nanmean(segment[:140])) < 0.2 and np.isnan(modulated[150])
    assert not np.allclose(modulated, segment, equal_nan=True)
    for _ in range(10):
        clipped = censor(segment, 140, rng)
        low, high = np.nanmin(clipped) > np.nanmin(segment), np.nanmax(clipped) < np.nanmax(segment)
        assert low != high and np.isnan(clipped[150])


def test_mixing_two_series_blends_them_on_a_common_scale_and_unknown_stays_unknown():
    a, b = np.full(100, 2.0), np.full(100, 400.0)
    b[90] = np.nan
    mixed = mix(a, b, 80, np.random.default_rng(0))
    assert np.allclose(np.delete(mixed, 90), 1.0) and np.isnan(mixed[90])
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    assert augmented([a, b], 80, V1, rng)[1] is b and rng.bit_generator.state == state


def test_synthetic_series_are_seeded_varied_in_length_and_finite():
    first = [Synthetic(800).draw(np.random.default_rng(5)) for _ in range(2)]
    assert np.array_equal(*first)
    rng = np.random.default_rng(0)
    drawn = [Synthetic(800).draw(rng) for _ in range(200)]
    lengths = [len(d) for d in drawn]
    assert min(lengths) >= MIN_LENGTH and max(lengths) <= 800 and len(set(lengths)) > 100
    assert all(np.isfinite(d).all() for d in drawn) and np.ptp([np.log(d.std() + 1e-9) for d in drawn]) > 3


def test_a_mixed_pick_draws_its_share_from_the_synthetic_series():
    from byosm.series.pretrain import mixed

    own = [(np.full(1000, 1.0), 48)]
    pick = mixed(Synthetic(300), lambda rng, n: [own[0]] * n, 0.5)
    chosen = pick(np.random.default_rng(0), 400)
    synthetic = sum(len(h) <= 300 for h, _ in chosen)
    assert len(chosen) == 400 and 150 < synthetic < 250


@pytest.fixture
def tiny_v2(tmp_path, monkeypatch):
    import byosm.series.patch as patch
    import byosm.series.patch_net as patch_net
    import byosm.series.pretrain as pretrain

    monkeypatch.setenv("BYOSM_CACHE", str(tmp_path))
    monkeypatch.setitem(patch_net.SIZES, "tiny", TINY)
    monkeypatch.setattr(pretrain, "_training_portions", lambda log: [(_wave(seed=i), MAX_HORIZON) for i in range(3)])
    for module in (pretrain, patch):
        monkeypatch.setattr(module, "_device", lambda: "cpu")
        monkeypatch.setattr(module, "BATCH", 8)
    monkeypatch.setattr(patch, "FINETUNE_STEPS", 2)
    return dataclasses.replace(V2, sizes=("tiny",))


def test_v2_pretraining_runs_with_no_downloaded_corpus_and_its_fine_tuned_model_serves_sorted_quantiles(tiny_v2):
    from byosm.series.patch import PretrainedPatch, missing_size, pretrained
    from byosm.series.pretrain import pretrain

    lines = []
    assert missing_size(tiny_v2) == "tiny" and pretrained(tiny_v2) is None
    path = pretrain(lines.append, 3, "tiny", tiny_v2)
    assert path.name == "patch-tiny-v2.pt" and pretrained(tiny_v2) == ("tiny-v2", TINY)
    assert missing_size(tiny_v2) is None
    assert any("synthetic" in line for line in lines) and torch.load(path)["head.weight"].shape[0] == 21 * MAX_HORIZON
    model = PretrainedPatch(recipe=tiny_v2)
    model.fit([Group([_wave(seed=i) for i in range(2)], 24, 30)])
    assert model.name == "patch transformer, pretrained v2"
    out = model.predict_many([_wave()[:-30], np.full(10, np.nan)], 24, 30)
    assert all(o.shape == (len(QUANTILES), 30) and np.all(np.diff(o, axis=0) >= 0) for o in out)
