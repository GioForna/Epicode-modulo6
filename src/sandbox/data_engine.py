"""Data Engine: preprocessing dei dati tabellari.

Pipeline:
1. gestione dei valori nulli (mediana per le numeriche, moda per le categoriche);
2. normalizzazione delle numeriche in [0, 1] con MinMaxScaler;
3. One-Hot Encoding delle categoriche.

Il preprocessor e' reversibile: ``inverse_transform`` riporta l'output del
decoder nello spazio originale (scala, tipi interi, categorie).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Una colonna numerica con pochi valori distinti (es. 0/1, livelli) viene
# trattata come categorica se il valore distinto e' sotto questa soglia.
MAX_CATEGORIES_FOR_NUMERIC = 2


@dataclass
class ColumnInfo:
    name: str
    kind: str  # "numeric" | "categorical"
    dtype: str
    is_integer: bool = False
    categories: list = field(default_factory=list)


class DataPreprocessor:
    """Trasforma un DataFrame in una matrice float32 adatta al VAE e viceversa."""

    def __init__(self, categorical_columns: list[str] | None = None):
        self.user_categorical = categorical_columns
        self.columns: list[ColumnInfo] = []
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.fill_values: dict = {}
        self.scaler: MinMaxScaler | None = None
        self.encoder: OneHotEncoder | None = None
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def _detect_types(self, df: pd.DataFrame) -> None:
        self.columns, self.numeric_cols, self.categorical_cols = [], [], []
        for col in df.columns:
            series = df[col]
            is_cat = (
                col in (self.user_categorical or [])
                or not pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or series.dropna().nunique() <= MAX_CATEGORIES_FOR_NUMERIC
            )
            if is_cat:
                self.categorical_cols.append(col)
                self.columns.append(ColumnInfo(col, "categorical", str(series.dtype)))
            else:
                non_null = series.dropna()
                is_int = bool(np.all(np.isclose(non_null, np.round(non_null))))
                self.numeric_cols.append(col)
                self.columns.append(
                    ColumnInfo(col, "numeric", str(series.dtype), is_integer=is_int)
                )

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        if df.empty:
            raise ValueError("Il dataset e' vuoto.")
        self._detect_types(df)

        for col in self.numeric_cols:
            median = df[col].median()
            self.fill_values[col] = 0.0 if pd.isna(median) else float(median)
        for col in self.categorical_cols:
            mode = df[col].mode(dropna=True)
            self.fill_values[col] = mode.iloc[0] if not mode.empty else "missing"

        clean = self._impute(df)

        if self.numeric_cols:
            self.scaler = MinMaxScaler().fit(clean[self.numeric_cols].to_numpy(float))
        if self.categorical_cols:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.encoder.fit(clean[self.categorical_cols].astype(str))
            for info, cats in zip(
                [c for c in self.columns if c.kind == "categorical"],
                self.encoder.categories_,
            ):
                info.categories = list(cats)

        self.fitted = True
        return self

    # ------------------------------------------------------------ transform
    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, value in self.fill_values.items():
            out[col] = out[col].fillna(value)
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        clean = self._impute(df[[c.name for c in self.columns]])
        parts = []
        if self.numeric_cols:
            parts.append(self.scaler.transform(clean[self.numeric_cols].to_numpy(float)))
        if self.categorical_cols:
            parts.append(self.encoder.transform(clean[self.categorical_cols].astype(str)))
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Riporta l'output del decoder nel formato del dataset originale."""
        self._check_fitted()
        data = np.asarray(data, dtype=np.float64)
        n_num = len(self.numeric_cols)
        result: dict[str, np.ndarray] = {}

        if n_num:
            num = np.clip(data[:, :n_num], 0.0, 1.0)
            num = self.scaler.inverse_transform(num)
            for i, col in enumerate(self.numeric_cols):
                result[col] = num[:, i]

        offset = n_num
        for info in (c for c in self.columns if c.kind == "categorical"):
            k = len(info.categories)
            block = data[:, offset : offset + k]
            result[info.name] = np.array(info.categories, dtype=object)[block.argmax(axis=1)]
            offset += k

        df = pd.DataFrame(result)[[c.name for c in self.columns]]
        return self._restore_dtypes(df)

    def _restore_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for info in self.columns:
            if info.kind == "numeric":
                if info.is_integer:
                    df[info.name] = np.round(df[info.name]).astype("int64")
            else:
                # Le categorie sono state codificate come stringhe: se in origine
                # la colonna era numerica (es. 0/1) riconverto il tipo.
                try:
                    df[info.name] = df[info.name].astype(info.dtype)
                except (TypeError, ValueError):
                    df[info.name] = df[info.name].astype(str)
        return df

    # -------------------------------------------------------------- utility
    @property
    def output_dim(self) -> int:
        self._check_fitted()
        n_cat = sum(len(c.categories) for c in self.columns if c.kind == "categorical")
        return len(self.numeric_cols) + n_cat

    def schema(self) -> list[dict]:
        """Descrizione delle colonne, usata anche nel prompt dell'LLM."""
        return [
            {
                "name": c.name,
                "kind": c.kind,
                "dtype": "int" if c.is_integer else ("float" if c.kind == "numeric" else "category"),
                "categories": [str(x) for x in c.categories],
            }
            for c in self.columns
        ]

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Il preprocessor non e' ancora stato addestrato (fit).")

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "DataPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
