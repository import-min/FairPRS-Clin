"""Tests for FairPRS-Clin — io, evaluate, and equity modules."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fairprs_clin.io import load_groups, load_scores
from fairprs_clin.evaluate import evaluate_scores, standardize_scores
from fairprs_clin.equity import (
    pairwise_smds,
    ks_tests,
    bootstrap_group_stats,
    flagging_disparity,
    sensitivity_curve,
    equalized_cutoffs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small DataFrame with two clearly separated groups."""
    rng = np.random.default_rng(0)
    eur = rng.normal(1.0, 0.8, 30)
    afr = rng.normal(-0.5, 0.9, 25)
    scores = np.concatenate([eur, afr])
    groups = ["EUR"] * 30 + ["AFR"] * 25
    df = pd.DataFrame({"IID": [f"S{i}" for i in range(55)], "SCORE": scores, "group": groups})
    df["SCORE_STD"] = (df["SCORE"] - df["SCORE"].mean()) / df["SCORE"].std(ddof=0)
    return df


# ── IO tests ──────────────────────────────────────────────────────────────

def test_load_groups_tsv(tmp_path):
    p = tmp_path / "groups.tsv"
    p.write_text("IID\tgroup\nA\tEUR\nB\tSAS\n")
    df = load_groups(p)
    assert set(df.columns) == {"IID", "group"}
    assert df.shape[0] == 2
    assert df["group"].tolist() == ["EUR", "SAS"]


def test_load_groups_alt_columns(tmp_path):
    """Flexible column naming: 'sample' and 'ancestry' should also work."""
    p = tmp_path / "groups.tsv"
    p.write_text("sample\tancestry\nA\tEUR\nB\tAFR\n")
    df = load_groups(p)
    assert set(df.columns) == {"IID", "group"}


def test_load_scores_csv(tmp_path):
    p = tmp_path / "scores.csv"
    p.write_text("IID,SCORE\nA,1.2\nB,3.4\nC,nan\n")
    df = load_scores(p, score_column="SCORE")
    # NaN row should be dropped
    assert df.shape[0] == 2


def test_load_scores_sscore(tmp_path):
    """PLINK2 .sscore format: whitespace-separated, SCORE1_SUM column."""
    p = tmp_path / "out.sscore"
    p.write_text("#FID\tIID\tNMISS_ALLELE_CT\tSCORE1_SUM\nFAM\tA\t100\t0.42\nFAM\tB\t100\t-0.31\n")
    df = load_scores(p)
    assert df.shape[0] == 2
    assert "SCORE" in df.columns


# ── Standardization tests ─────────────────────────────────────────────────

def test_standardize_global(sample_df):
    df = standardize_scores(sample_df[["IID","SCORE","group"]].copy(), how="global")
    assert abs(df["SCORE_STD"].mean()) < 1e-9
    assert abs(df["SCORE_STD"].std(ddof=0) - 1.0) < 1e-9


def test_standardize_within_group(sample_df):
    df = standardize_scores(sample_df[["IID","SCORE","group"]].copy(), how="within_group")
    for grp in df["group"].unique():
        sub = df.loc[df["group"] == grp, "SCORE_STD"]
        assert abs(sub.mean()) < 1e-9


def test_standardize_none(sample_df):
    df = standardize_scores(sample_df[["IID","SCORE","group"]].copy(), how="none")
    assert (df["SCORE_STD"] == df["SCORE"]).all()


# ── Evaluate integration test ─────────────────────────────────────────────

def test_evaluate_full_pipeline(tmp_path):
    scores = tmp_path / "scores.csv"
    scores.write_text("IID,SCORE\nA,0.0\nB,1.0\nC,2.0\nD,-1.0\nE,1.5\n")
    groups = tmp_path / "groups.tsv"
    groups.write_text("IID\tgroup\nA\tEUR\nB\tEUR\nC\tEUR\nD\tAFR\nE\tAFR\n")
    out = tmp_path / "out"

    meta = evaluate_scores(
        scores, groups, out,
        cutoff=None, cutoff_percentile=80,
        standardize="global", score_column="SCORE",
        n_boot=50,
    )

    # Core outputs
    assert (out / "tables" / "summary_by_group.tsv").exists()
    assert (out / "figures" / "score_distributions_kde.png").exists()

    # New equity outputs
    assert (out / "tables" / "pairwise_smds.tsv").exists()
    assert (out / "tables" / "ks_tests.tsv").exists()
    assert (out / "tables" / "bootstrap_stats.tsv").exists()
    assert (out / "tables" / "sensitivity_curve.tsv").exists()
    assert (out / "figures" / "sensitivity_curve.png").exists()
    assert (out / "figures" / "disparity_curve.png").exists()
    assert (out / "figures" / "smd_heatmap.png").exists()

    assert meta["n_samples_with_groups"] == 5
    assert "equity" in meta


# ── Equity module tests ───────────────────────────────────────────────────

def test_pairwise_smds_shape(sample_df):
    result = pairwise_smds(sample_df)
    # 2 groups → 1 pair
    assert len(result) == 1
    assert "smd" in result.columns
    assert "abs_smd" in result.columns


def test_pairwise_smds_separated_groups(sample_df):
    """EUR mean > AFR mean → SMD should be large."""
    result = pairwise_smds(sample_df)
    assert result["abs_smd"].iloc[0] > 0.5


def test_ks_tests_significant(sample_df):
    result = ks_tests(sample_df)
    assert len(result) == 1
    # Groups are well-separated so should be significant
    assert result["significant_p05"].iloc[0] == True


def test_bootstrap_returns_cis(sample_df):
    cutoff = sample_df["SCORE_STD"].quantile(0.80)
    result = bootstrap_group_stats(sample_df, cutoff_value=cutoff, n_boot=100)
    assert "mean_ci_lo" in result.columns
    assert "mean_ci_hi" in result.columns
    assert "flag_ci_lo" in result.columns
    # CI lo < mean < CI hi for each group
    for _, row in result.iterrows():
        assert row["mean_ci_lo"] <= row["mean"]
        assert row["mean"] <= row["mean_ci_hi"]


def test_flagging_disparity_ratio(sample_df):
    cutoff = sample_df["SCORE_STD"].quantile(0.80)
    result = flagging_disparity(sample_df, cutoff_value=cutoff)
    assert "disparity_ratio" in result
    assert result["disparity_ratio"] >= 1.0  # by definition max/min >= 1
    assert "max_flagging_group" in result
    assert "absolute_disparity_pp" in result


def test_sensitivity_curve_shape(sample_df):
    result = sensitivity_curve(sample_df, percentiles=list(range(70, 95)))
    groups = sample_df["group"].unique()
    # Should have one row per (percentile, group) combination
    assert len(result) == 25 * len(groups)
    assert "flagging_rate" in result.columns
    assert "disparity_ratio" in result.columns


def test_sensitivity_curve_monotone(sample_df):
    """Flagging rate must decrease (or stay flat) as cutoff percentile increases."""
    result = sensitivity_curve(sample_df, percentiles=list(range(50, 99)))
    for grp in sample_df["group"].unique():
        sub = result[result["group"] == grp].sort_values("percentile")
        rates = sub["flagging_rate"].values
        # Allow floating point tolerance
        assert all(rates[i] >= rates[i + 1] - 1e-9 for i in range(len(rates) - 1))


def test_equalized_cutoffs(sample_df):
    target = 0.15
    result = equalized_cutoffs(sample_df, target_flagging_rate=target)
    assert len(result) == sample_df["group"].nunique()
    for _, row in result.iterrows():
        # Achieved rate should be close to target (within one discrete step)
        assert abs(row["achieved_flagging_rate"] - target) < 0.15


def test_equalized_cutoffs_different_per_group(sample_df):
    """Groups with different distributions must get different equalized cutoffs."""
    result = equalized_cutoffs(sample_df, target_flagging_rate=0.10)
    cutoffs = result.set_index("group")["equalized_cutoff"]
    # EUR (higher mean) should need a higher cutoff than AFR
    assert cutoffs["EUR"] > cutoffs["AFR"]
