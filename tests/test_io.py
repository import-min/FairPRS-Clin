from pathlib import Path
import pandas as pd
from fairprs_clin.io import load_groups, load_scores

def test_load_groups(tmp_path: Path):
    p = tmp_path / "groups.tsv"
    p.write_text("IID\tgroup\nA\tEUR\nB\tSAS\n")
    df = load_groups(p)
    assert set(df.columns) == {"IID","group"}
    assert df.shape[0] == 2

def test_load_scores_csv(tmp_path: Path):
    p = tmp_path / "scores.csv"
    p.write_text("IID,SCORE\nA,1.2\nB,3.4\n")
    df = load_scores(p, score_column="SCORE")
    assert df.shape[0] == 2

def test_load_scores_plink2_hash_iid(tmp_path):
    p = tmp_path / "real.sscore"
    p.write_text(
        "#IID\tALLELE_CT\tNAMED_ALLELE_DOSAGE_SUM\tsee_AVG\tsee_SUM\n"
        "HG00096\t335626\t136493\t-0.00717169\t-2407.01\n"
        "HG00097\t335626\t136787\t-0.00728591\t-2445.34\n"
    )
    df = load_scores(p)
    assert list(df.columns) == ["IID", "SCORE"]
    assert df["IID"].tolist() == ["HG00096", "HG00097"]
    assert len(df) == 2
