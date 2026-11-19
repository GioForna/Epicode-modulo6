"""Addestramento, salvataggio e caricamento del VAE."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .data_engine import DataPreprocessor
from .vae import TabularVAE, vae_loss

ProgressCallback = Callable[[int, int, dict], None]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyntheticDataModel:
    """Raggruppa preprocessor + VAE: e' l'oggetto che viene addestrato,
    salvato su disco e usato per generare i dati sintetici."""

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self.preprocessor: DataPreprocessor | None = None
        self.model: TabularVAE | None = None
        self.history: list[dict] = []
        self.device = get_device()

    # --------------------------------------------------------------- train
    def fit(
        self,
        df: pd.DataFrame,
        categorical_columns: list[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> "SyntheticDataModel":
        cfg = self.config
        set_seed(cfg.seed)

        self.preprocessor = DataPreprocessor(categorical_columns)
        x = torch.tensor(self.preprocessor.fit_transform(df))

        self.model = TabularVAE(
            input_dim=x.shape[1],
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
        ).to(self.device)

        # drop_last evita batch da 1 elemento, che darebbero errore con BatchNorm
        batch_size = min(cfg.batch_size, len(x))
        loader = DataLoader(
            TensorDataset(x),
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(x) > batch_size,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=20
        )

        self.history = []
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            tot = rec = kld = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model(batch)
                # KL annealing: il peso della KL cresce nelle prime epoche,
                # cosi' il modello impara prima a ricostruire e non collassa
                # subito sulla prior.
                beta = cfg.beta * min(1.0, epoch / max(1, cfg.epochs // 4))
                loss, r, k = vae_loss(recon, batch, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                tot, rec, kld = tot + loss.item(), rec + r.item(), kld + k.item()
                n_batches += 1

            metrics = {
                "epoch": epoch,
                "loss": tot / n_batches,
                "recon_loss": rec / n_batches,
                "kl_loss": kld / n_batches,
            }
            scheduler.step(metrics["loss"])
            self.history.append(metrics)
            if progress:
                progress(epoch, cfg.epochs, metrics)

        return self

    # ------------------------------------------------------------ generate
    def generate(self, n_samples: int, seed: int | None = None) -> pd.DataFrame:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Il modello non e' stato addestrato.")
        if seed is not None:
            torch.manual_seed(seed)
        raw = self.model.sample(n_samples, device=self.device).cpu().numpy()
        return self.preprocessor.inverse_transform(raw)

    # ---------------------------------------------------------- save/load
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "vae.pt")
        self.preprocessor.save(directory / "preprocessor.pkl")
        meta = {
            "model": self.model.config(),
            "train_config": {**asdict(self.config), "hidden_dims": list(self.config.hidden_dims)},
            "history": self.history,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "SyntheticDataModel":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        tc = meta["train_config"]
        tc["hidden_dims"] = tuple(tc["hidden_dims"])
        obj = cls(TrainConfig(**tc))
        obj.preprocessor = DataPreprocessor.load(directory / "preprocessor.pkl")
        mc = meta["model"]
        obj.model = TabularVAE(mc["input_dim"], mc["latent_dim"], tuple(mc["hidden_dims"]))
        obj.model.load_state_dict(torch.load(directory / "vae.pt", map_location=obj.device))
        obj.model.to(obj.device).eval()
        obj.history = meta.get("history", [])
        return obj
