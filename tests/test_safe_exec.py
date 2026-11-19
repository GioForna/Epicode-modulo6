import pandas as pd
import pytest

from sandbox.safe_exec import UnsafeCodeError, execute_code, validate_code

DF = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})


@pytest.mark.parametrize("code", [
    "import os\nos.system('ls')",
    "from subprocess import run",
    "open('/etc/passwd').read()",
    "df.__class__.__bases__",
    "eval('1+1')",
    "df.to_csv('x.csv')",
    "pd.read_csv('/etc/passwd')",
    "while True:\n    pass",
    "__import__('os')",
])
def test_rejects_unsafe_code(code):
    with pytest.raises(UnsafeCodeError):
        validate_code(code)


def test_executes_analysis_and_plot():
    code = (
        "result = df['a'].corr(df['b'])\n"
        "fig, ax = plt.subplots()\n"
        "ax.scatter(df['a'], df['b'])\n"
        "print('ok')"
    )
    res = execute_code(code, DF)
    assert res.success, res.error
    assert float(res.result) == pytest.approx(1.0)
    assert len(res.images) == 1
    assert "ok" in res.stdout


def test_runtime_error_is_reported():
    res = execute_code("result = df['colonna_inesistente'].mean()", DF)
    assert not res.success
    assert "KeyError" in res.error


def test_original_dataframe_not_modified():
    df = DF.copy()
    execute_code("df['a'] = 0", df)
    assert df["a"].tolist() == [1, 2, 3, 4]


def test_timeout():
    res = execute_code("x = sum(i for i in range(10**12))", DF, timeout=2)
    assert not res.success and "timeout" in res.error
