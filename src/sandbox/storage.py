"""Gestione su disco dei dataset caricati, dei modelli e dei dati sintetici.

Struttura:
    storage/<dataset_id>/
        original.csv     dataset caricato dall'utente
        meta.json        info sul dataset e stato del training
        model/           pesi del VAE + preprocessor
        synthetic.csv    ultimo dataset sintetico generato
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import STORAGE_DIR

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_lock = threading.Lock()


class DatasetNotFound(KeyError):
    pass


class DatasetStore:
    def __init__(self, root: Path | str = STORAGE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- paths
    def path(self, dataset_id: str) -> Path:
        # Validazione dell'id: evita path traversal (es. "../../etc")
        if not _ID_RE.match(dataset_id or ""):
            raise DatasetNotFound(dataset_id)
        p = self.root / dataset_id
        if not p.exists():
            raise DatasetNotFound(dataset_id)
        return p

    def model_dir(self, dataset_id: str) -> Path:
        return self.path(dataset_id) / "model"

    # ----------------------------------------------------------------- crud
    def create(self, df: pd.DataFrame, filename: str) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        p = self.root / dataset_id
        p.mkdir(parents=True)
        df.to_csv(p / "original.csv", index=False)
        self._write_meta(dataset_id, {
            "dataset_id": dataset_id,
            "filename": filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": list(map(str, df.columns)),
            "training": {"status": "not_trained"},
        })
        return dataset_id

    def list(self) -> list[dict]:
        items = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir() and (p / "meta.json").exists():
                items.append(json.loads((p / "meta.json").read_text()))
        return sorted(items, key=lambda m: m.get("uploaded_at", ""), reverse=True)

    def meta(self, dataset_id: str) -> dict:
        return json.loads((self.path(dataset_id) / "meta.json").read_text())

    def update_meta(self, dataset_id: str, **fields) -> dict:
        with _lock:
            meta = self.meta(dataset_id)
            meta.update(fields)
            self._write_meta(dataset_id, meta)
        return meta

    def _write_meta(self, dataset_id: str, meta: dict) -> None:
        target = self.root / dataset_id / "meta.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        tmp.replace(target)

    def load_original(self, dataset_id: str) -> pd.DataFrame:
        return pd.read_csv(self.path(dataset_id) / "original.csv")

    def save_synthetic(self, dataset_id: str, df: pd.DataFrame) -> None:
        df.to_csv(self.path(dataset_id) / "synthetic.csv", index=False)

    def load_synthetic(self, dataset_id: str) -> pd.DataFrame:
        f = self.path(dataset_id) / "synthetic.csv"
        if not f.exists():
            raise FileNotFoundError("Nessun dataset sintetico generato per questo dataset.")
        return pd.read_csv(f)

    def has_model(self, dataset_id: str) -> bool:
        return (self.model_dir(dataset_id) / "vae.pt").exists()
