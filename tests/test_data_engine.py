import numpy as np

from sandbox.data_engine import DataPreprocessor


def test_types_detection(mixed_df):
    pre = DataPreprocessor().fit(mixed_df)
    assert pre.numeric_cols == ["age", "income", "savings"]
    assert pre.categorical_cols == ["city", "default"]
    assert pre.columns[0].is_integer and not pre.columns[1].is_integer


def test_transform_range_and_shape(mixed_df):
    pre = DataPreprocessor()
    x = pre.fit_transform(mixed_df)
    # 3 numeriche + 3 citta' + 2 classi di default
    assert x.shape == (len(mixed_df), 8) == (len(mixed_df), pre.output_dim)
    assert not np.isnan(x).any()
    assert x.min() >= 0 and x.max() <= 1


def test_inverse_transform_roundtrip(mixed_df):
    pre = DataPreprocessor()
    x = pre.fit_transform(mixed_df)
    back = pre.inverse_transform(x)
    assert list(back.columns) == list(mixed_df.columns)
    assert back["age"].dtype.kind == "i"
    assert back["default"].dtype.kind == "i"
    np.testing.assert_allclose(back["age"], mixed_df["age"])
    mask = mixed_df["income"].notna()
    np.testing.assert_allclose(back.loc[mask, "income"], mixed_df.loc[mask, "income"], rtol=1e-5)
    assert set(back["city"]) <= {"Milano", "Roma", "Napoli"}


def test_user_categorical(mixed_df):
    pre = DataPreprocessor(categorical_columns=["age"]).fit(mixed_df)
    assert "age" in pre.categorical_cols


def test_save_load(tmp_path, mixed_df):
    pre = DataPreprocessor().fit(mixed_df)
    pre.save(tmp_path / "p.pkl")
    loaded = DataPreprocessor.load(tmp_path / "p.pkl")
    np.testing.assert_array_equal(pre.transform(mixed_df), loaded.transform(mixed_df))
