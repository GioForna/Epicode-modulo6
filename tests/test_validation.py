from sandbox.data_engine import DataPreprocessor
from sandbox.validation import comparison_table, full_report, privacy_report


def test_privacy_detects_copies(mixed_df):
    copy = mixed_df.dropna().head(30)
    rep = privacy_report(mixed_df, copy, DataPreprocessor().fit(mixed_df))
    assert rep["exact_matches"] == 30
    assert rep["passed"] is False


def test_full_report_identical_data(mixed_df):
    rep = full_report(mixed_df, mixed_df)
    assert rep["fidelity_score"] > 0.99
    assert rep["correlation"]["mean_abs_diff"] < 1e-9
    assert {d["column"] for d in rep["distribution"]} == set(mixed_df.columns)


def test_comparison_table(mixed_df):
    table = comparison_table(mixed_df, mixed_df)
    assert ("mean", "Reale") in table.columns and ("mean", "Sintetico") in table.columns
