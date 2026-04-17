"""Tests for novel FairPRS-Clin algorithms: APS, BGR, RCFT."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fairprs_clin.portability import (
    aps_distributional, aps_clinical, bootstrap_aps, _harmonic_mean,
)
from fairprs_clin.recalibration import (
    bayesian_group_recalibration, evaluate_recalibration,
    _fit_group_map, _shrinkage_lambda,
)
from fairprs_clin.equity import resource_constrained_fair_threshold


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def well_separated_df():
    """EUR scores much higher than AFR — low portability."""
    rng = np.random.default_rng(1)
    eur = rng.normal(1.5, 0.8, 50)
    afr = rng.normal(-1.0, 0.8, 40)
    eas = rng.normal(-0.8, 0.8, 35)
    scores = np.concatenate([eur, afr, eas])
    groups = ["EUR"] * 50 + ["AFR"] * 40 + ["EAS"] * 35
    df = pd.DataFrame({"IID": [f"S{i}" for i in range(125)],
                       "SCORE": scores, "group": groups})
    df["SCORE_STD"] = (df["SCORE"] - df["SCORE"].mean()) / df["SCORE"].std(ddof=0)
    return df


@pytest.fixture
def portable_df():
    """All groups have same distribution — high portability."""
    rng = np.random.default_rng(2)
    n = 40
    scores = np.concatenate([rng.normal(0.0, 1.0, n) for _ in range(3)])
    groups = ["EUR"] * n + ["AFR"] * n + ["EAS"] * n
    df = pd.DataFrame({"IID": [f"S{i}" for i in range(3 * n)],
                       "SCORE": scores, "group": groups})
    df["SCORE_STD"] = (df["SCORE"] - df["SCORE"].mean()) / df["SCORE"].std(ddof=0)
    return df


@pytest.fixture
def df_with_outcomes(well_separated_df):
    """Add synthetic binary outcomes correlated with score."""
    rng = np.random.default_rng(42)
    df = well_separated_df.copy()
    logits = -1.0 + 0.5 * df["SCORE_STD"].values
    probs = 1 / (1 + np.exp(-logits))
    df["outcome"] = (rng.uniform(0, 1, len(df)) < probs).astype(int)
    return df


# ── APS tests ─────────────────────────────────────────────────────────────

def test_aps_range(well_separated_df, portable_df):
    """APS must be in [0, 1]."""
    aps_low = aps_distributional(well_separated_df)
    aps_high = aps_distributional(portable_df)
    assert 0.0 <= aps_low["aps"] <= 1.0
    assert 0.0 <= aps_high["aps"] <= 1.0


def test_aps_separated_lower_than_portable(well_separated_df, portable_df):
    """Well-separated groups → lower APS than portable groups."""
    aps_low = aps_distributional(well_separated_df)["aps"]
    aps_high = aps_distributional(portable_df)["aps"]
    assert aps_low < aps_high, f"Expected aps_low ({aps_low}) < aps_high ({aps_high})"


def test_aps_with_cutoff(well_separated_df):
    """APS with flagging component should be <= APS without."""
    cutoff = float(np.percentile(well_separated_df["SCORE_STD"], 90))
    aps_no_cutoff = aps_distributional(well_separated_df)["aps"]
    aps_with_cutoff = aps_distributional(well_separated_df, cutoff_value=cutoff)["aps"]
    # With cutoff, flagging disparity adds a penalty → should be ≤ without
    assert aps_with_cutoff <= aps_no_cutoff + 0.01  # small tolerance for harmonic mean


def test_aps_distributional_components(well_separated_df):
    """All expected components present in output."""
    result = aps_distributional(well_separated_df)
    assert "aps" in result
    assert "aps_distributional" in result
    assert "max_wasserstein_distance" in result
    assert "per_group_wasserstein" in result
    assert "interpretation" in result


def test_aps_bootstrap_ci_ordered(well_separated_df):
    """Bootstrap CI must satisfy lo ≤ point ≤ hi."""
    result = bootstrap_aps(well_separated_df, n_boot=100)
    if not np.isnan(result["aps_point"]):
        assert result["aps_ci_lo"] <= result["aps_point"]
        assert result["aps_point"] <= result["aps_ci_hi"]


def test_aps_clinical_requires_both_dfs():
    """aps_clinical should return nan for empty DataFrames."""
    empty = pd.DataFrame(columns=["group", "auc"])
    empty_cal = pd.DataFrame(columns=["group", "calibration_slope"])
    result = aps_clinical(empty, empty_cal)
    assert np.isnan(result["aps"])


def test_harmonic_mean_basic():
    assert abs(_harmonic_mean([1.0, 1.0]) - 1.0) < 1e-9
    assert abs(_harmonic_mean([0.5, 1.0]) - (2 / (1 / 0.5 + 1 / 1.0))) < 1e-9
    assert np.isnan(_harmonic_mean([]))
    assert np.isnan(_harmonic_mean([float("nan")]))
    # Zero dominates harmonic mean — APS=0 means worst portability, not missing
    assert _harmonic_mean([0.0]) == 0.0
    assert _harmonic_mean([0.0, 1.0]) == 0.0


# ── BGR tests ─────────────────────────────────────────────────────────────

def test_bgr_returns_recalibrated_scores(df_with_outcomes):
    """BGR must add SCORE_BGR column to output DataFrame."""
    df_recal, params_df = bayesian_group_recalibration(df_with_outcomes)
    assert "SCORE_BGR" in df_recal.columns
    assert df_recal["SCORE_BGR"].notna().all()


def test_bgr_params_per_group(df_with_outcomes):
    """One row per group in params DataFrame."""
    _, params_df = bayesian_group_recalibration(df_with_outcomes)
    n_groups = df_with_outcomes["group"].nunique()
    assert len(params_df) == n_groups


def test_bgr_shrinkage_small_n():
    """Small n → high shrinkage (close to 1.0)."""
    lam = _shrinkage_lambda(5, n_threshold=100)
    shrinkage = lam / (1.0 + lam)
    assert shrinkage > 0.9, f"Expected high shrinkage for n=5, got {shrinkage:.3f}"


def test_bgr_shrinkage_large_n():
    """Large n → low shrinkage (close to 0.0)."""
    lam = _shrinkage_lambda(10000, n_threshold=100)
    shrinkage = lam / (1.0 + lam)
    assert shrinkage < 0.05, f"Expected low shrinkage for n=10000, got {shrinkage:.3f}"


def test_bgr_global_fallback_small_cases(df_with_outcomes):
    """Groups with < min_cases should use global_fallback method."""
    # AFR in well_separated_df has few cases in toy data
    _, params_df = bayesian_group_recalibration(df_with_outcomes, min_cases=100)
    # With min_cases=100, all groups should fall back to global
    assert (params_df["recal_method"] == "global_fallback").all()


def test_bgr_map_fit_converges():
    """MAP optimization should converge for well-defined problem."""
    rng = np.random.default_rng(7)
    s = rng.normal(0, 1, 100)
    y = (rng.normal(0, 1, 100) + 0.5 * s > 0).astype(float)
    alpha, beta, converged = _fit_group_map(s, y, 0.0, 1.0, lambda_reg=1.0)
    assert converged
    assert not np.isnan(alpha)
    assert not np.isnan(beta)


def test_bgr_evaluation_returns_deltas(df_with_outcomes):
    """evaluate_recalibration returns auc_delta per group."""
    df_recal, _ = bayesian_group_recalibration(df_with_outcomes)
    eval_df = evaluate_recalibration(df_with_outcomes, df_recal)
    assert "auc_delta" in eval_df.columns
    assert len(eval_df) == df_with_outcomes["group"].nunique()


def test_bgr_insufficient_data_raises():
    """BGR with too few cases globally should raise ValueError."""
    df = pd.DataFrame({
        "IID": ["A", "B", "C"],
        "SCORE": [0.1, 0.2, 0.3],
        "group": ["EUR", "EUR", "AFR"],
        "SCORE_STD": [0.1, 0.2, 0.3],
        "outcome": [1, 0, 0],
    })
    with pytest.raises(ValueError, match="Insufficient data"):
        bayesian_group_recalibration(df)


# ── RCFT tests ────────────────────────────────────────────────────────────

def test_rcft_budget_satisfied(well_separated_df):
    """RCFT actual budget used should be ≤ budget + tolerance."""
    budget = 0.10
    result = resource_constrained_fair_threshold(well_separated_df, budget=budget)
    actual = result["actual_budget_used"].iloc[0]
    assert actual <= budget * 1.1, f"Budget violated: {actual:.4f} > {budget}"


def test_rcft_returns_one_row_per_group(well_separated_df):
    """RCFT returns one row per ancestry group."""
    result = resource_constrained_fair_threshold(well_separated_df, budget=0.10)
    assert len(result) == well_separated_df["group"].nunique()


def test_rcft_reduces_disparity(well_separated_df):
    """RCFT disparity ratio should be ≤ naive disparity ratio."""
    result = resource_constrained_fair_threshold(well_separated_df, budget=0.10)
    dr_rcft = result["disparity_ratio_rcft"].iloc[0]
    dr_naive = result["disparity_ratio_naive"].iloc[0]
    if dr_naive is not None and not np.isnan(dr_naive) and not np.isinf(dr_naive):
        assert dr_rcft <= dr_naive + 0.1, (
            f"RCFT disparity ({dr_rcft:.2f}) should be ≤ naive ({dr_naive:.2f})"
        )


def test_rcft_all_fairness_criteria(well_separated_df):
    """All three fairness criteria should run without error."""
    for criterion in ["demographic_parity", "chebyshev", "variance"]:
        result = resource_constrained_fair_threshold(
            well_separated_df, budget=0.10, fairness_criterion=criterion
        )
        assert len(result) == well_separated_df["group"].nunique()
        assert "rcft_cutoff" in result.columns


def test_rcft_invalid_criterion_raises(well_separated_df):
    """Invalid fairness criterion should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown fairness_criterion"):
        resource_constrained_fair_threshold(
            well_separated_df, budget=0.10, fairness_criterion="invalid"
        )


def test_rcft_budget_zero_flags_nobody(well_separated_df):
    """Budget = 0 should result in near-zero flagging."""
    result = resource_constrained_fair_threshold(well_separated_df, budget=0.001)
    # With budget near 0, most groups should have near-zero flagging
    max_rate = result["rcft_flagging_rate"].max()
    assert max_rate < 0.15


# ── Integration test ──────────────────────────────────────────────────────

def test_full_pipeline_with_all_algorithms(tmp_path):
    """End-to-end integration test: all three novel algorithms run."""
    from fairprs_clin.evaluate import evaluate_scores

    scores = tmp_path / "scores.csv"
    rng = np.random.default_rng(99)
    iids = [f"S{i}" for i in range(80)]
    groups_list = ["EUR"] * 30 + ["AFR"] * 25 + ["EAS"] * 25
    raw_scores = np.concatenate([
        rng.normal(0.5, 1.0, 30),
        rng.normal(-0.3, 1.0, 25),
        rng.normal(-0.2, 1.0, 25),
    ])
    pd.DataFrame({"IID": iids, "SCORE": raw_scores}).to_csv(scores, index=False)

    groups = tmp_path / "groups.tsv"
    pd.DataFrame({"IID": iids, "group": groups_list}).to_csv(groups, sep="\t", index=False)

    outcomes = tmp_path / "outcomes.tsv"
    logits = -1.0 + 0.6 * raw_scores
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, 80) < probs).astype(int)
    pd.DataFrame({"IID": iids, "outcome": y}).to_csv(outcomes, sep="\t", index=False)

    out = tmp_path / "out"
    meta = evaluate_scores(
        scores, groups, out,
        cutoff=None, cutoff_percentile=85,
        standardize="global", score_column="SCORE",
        n_boot=50, equalize_target=0.10,
        outcomes_path=outcomes, outcome_column="outcome",
        run_bgr=True, bgr_n_threshold=50,
        rcft_budget=0.10,
    )

    # APS computed
    assert "aps_distributional" in meta
    assert meta["aps_distributional"]["aps_point"] is not None

    # RCFT computed
    assert (out / "tables" / "rcft_thresholds.tsv").exists()
    assert meta["rcft"]["budget"] == 0.10

    # BGR ran
    assert (out / "tables" / "bgr_parameters.tsv").exists()
    assert (out / "figures" / "bgr_parameters.png").exists()

    # Standard outputs
    assert (out / "tables" / "summary_by_group.tsv").exists()
    assert (out / "figures" / "aps_gauge.png").exists()
    assert (out / "figures" / "rcft_thresholds.png").exists()
