import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """Piccolo dataset misto: numeriche, intere, categoriche e valori nulli."""
    rng = np.random.default_rng(0)
    n = 200
    age = rng.integers(18, 80, n)
    income = rng.normal(30000, 8000, n).round(2)
    df = pd.DataFrame({
        "age": age,
        "income": income,
        "savings": (income * 0.1 + rng.normal(0, 500, n)).round(2),
        "city": rng.choice(["Milano", "Roma", "Napoli"], n),
        "default": rng.choice([0, 1], n),
    })
    df.loc[[3, 10, 50], "income"] = np.nan
    df.loc[[5, 7], "city"] = np.nan
    return df
