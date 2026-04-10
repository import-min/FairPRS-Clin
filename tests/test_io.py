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
