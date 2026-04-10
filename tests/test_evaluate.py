from pathlib import Path
from fairprs_clin.evaluate import evaluate_scores

def test_evaluate(tmp_path: Path):
    scores = tmp_path / "scores.csv"
    scores.write_text("IID,SCORE\nA,0.0\nB,1.0\nC,2.0\n")
    groups = tmp_path / "groups.tsv"
    groups.write_text("IID\tgroup\nA\tEUR\nB\tEUR\nC\tSAS\n")
    out = tmp_path / "out"
    meta = evaluate_scores(scores, groups, out, cutoff=None, cutoff_percentile=90, standardize="global", score_column="SCORE")
    assert (out / "tables" / "summary_by_group.tsv").exists()
    assert (out / "figures" / "score_distributions.png").exists()
    assert meta["n_samples_with_groups"] == 3
