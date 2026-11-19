"""Prepara il dataset Pima Indians Diabetes partendo dal CSV grezzo.

Il file originale non ha intestazione e codifica i valori mancanti con 0
in alcune colonne dove lo zero non ha senso clinico (glicemia, pressione,
BMI...). Qui aggiungo i nomi delle colonne e converto quegli zeri in NaN,
cosi' il Data Engine li gestisce come veri valori nulli.

Uso:
    python scripts/prepare_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pima-indians-diabetes.data.csv"
OUT = ROOT / "data" / "diabetes.csv"

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Colonne in cui 0 significa "dato non rilevato"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def main() -> None:
    df = pd.read_csv(RAW, header=None, names=COLUMNS)
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    # Outcome e' una variabile binaria: la tratto come categorica
    # cosi' viene codificata con One-Hot Encoding.
    df["Outcome"] = df["Outcome"].map({0: "No", 1: "Yes"})

    df.to_csv(OUT, index=False)
    print(f"Salvato {OUT} ({len(df)} righe)")
    print("Valori nulli per colonna:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
