"""Configurazione centralizzata del progetto (percorsi e parametri di default)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = Path(os.getenv("SANDBOX_STORAGE_DIR", ROOT_DIR / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_URL = os.getenv("SANDBOX_API_URL", "http://localhost:8000")


@dataclass
class TrainConfig:
    """Iperparametri dell'addestramento del VAE."""

    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-3
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (64, 32)
    # Peso della KL Divergence nella loss (beta-VAE). Con le feature in [0, 1]
    # la MSE e' molto piccola: con beta=1 la KL domina e il modello collassa
    # sulla media (dati sintetici con varianza quasi nulla). Con beta basso
    # le distribuzioni generate sono molto piu' fedeli (vedi README).
    beta: float = 0.01
    seed: int = 42
