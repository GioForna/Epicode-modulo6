"""Analytic Backend: API FastAPI per upload, training, generazione,
validazione e interrogazione in linguaggio naturale.

Avvio:
    uvicorn api.main:app --reload
Documentazione interattiva: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sandbox import __version__
from sandbox.config import TrainConfig
from sandbox.llm_analyst import NaturalLanguageAnalyst
from sandbox.storage import DatasetNotFound, DatasetStore
from sandbox.trainer import SyntheticDataModel
from sandbox.validation import comparison_table, full_report

MAX_UPLOAD_MB = 50

app = FastAPI(
    title="Synthetic Data Sandbox - VAE Edition",
    description="Caricamento di dataset sensibili, training di un VAE e generazione di dati sintetici anonimi.",
    version=__version__,
)

_store = DatasetStore()
_models: dict[str, SyntheticDataModel] = {}  # cache dei modelli caricati


def get_store() -> DatasetStore:
    return _store


# ------------------------------------------------------------------ schemi


class TrainRequest(BaseModel):
    dataset_id: str
    epochs: int = Field(300, ge=1, le=5000)
    batch_size: int = Field(64, ge=8, le=4096)
    learning_rate: float = Field(1e-3, gt=0, le=1)
    latent_dim: int = Field(8, ge=1, le=256)
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32], min_length=1, max_length=5)
    beta: float = Field(0.01, ge=0, le=10)
    categorical_columns: list[str] | None = None
    seed: int = 42


class ValidateRequest(BaseModel):
    dataset_id: str


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=2, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


# ------------------------------------------------------------------ helper


def _meta_or_404(store: DatasetStore, dataset_id: str) -> dict:
    try:
        return store.meta(dataset_id)
    except DatasetNotFound:
        raise HTTPException(404, f"Dataset '{dataset_id}' non trovato.") from None


def _get_model(store: DatasetStore, dataset_id: str) -> SyntheticDataModel:
    if dataset_id in _models:
        return _models[dataset_id]
    if not store.has_model(dataset_id):
        raise HTTPException(409, "Il modello non e' ancora stato addestrato: chiama prima /train.")
    model = SyntheticDataModel.load(store.model_dir(dataset_id))
    _models[dataset_id] = model
    return model


def _run_training(store: DatasetStore, req: TrainRequest) -> None:
    dataset_id = req.dataset_id
    cfg = TrainConfig(
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        latent_dim=req.latent_dim,
        hidden_dims=tuple(req.hidden_dims),
        beta=req.beta,
        seed=req.seed,
    )
    history: list[dict] = []

    def on_epoch(epoch: int, total: int, metrics: dict) -> None:
        history.append({k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()})
        # Aggiorno lo stato su disco ogni 10 epoche (e all'ultima)
        if epoch % 10 == 0 or epoch == total:
            store.update_meta(dataset_id, training={
                "status": "running", "epoch": epoch, "epochs": total, "history": history,
            })

    try:
        df = store.load_original(dataset_id)
        model = SyntheticDataModel(cfg).fit(df, req.categorical_columns, progress=on_epoch)
        model.save(store.model_dir(dataset_id))
        _models[dataset_id] = model
        store.update_meta(dataset_id, schema=model.preprocessor.schema(), training={
            "status": "completed",
            "epoch": cfg.epochs,
            "epochs": cfg.epochs,
            "history": history,
            "config": req.model_dump(exclude={"dataset_id"}),
        })
    except Exception as exc:  # noqa: BLE001
        store.update_meta(dataset_id, training={"status": "failed", "error": str(exc)})


# --------------------------------------------------------------- endpoint


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/upload")
async def upload(file: UploadFile = File(...), store: DatasetStore = Depends(get_store)) -> dict:
    """Carica un CSV e restituisce l'id del dataset con un'anteprima."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Sono accettati solo file .csv")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File troppo grande (max {MAX_UPLOAD_MB} MB).")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"CSV non valido: {exc}") from exc
    if df.empty or df.shape[1] < 2:
        raise HTTPException(400, "Il CSV deve contenere almeno una riga e due colonne.")

    dataset_id = store.create(df, file.filename)
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(v) for c, v in df.isna().sum().items()},
        "preview": df.head(10).astype(object).where(df.head(10).notna(), None).to_dict("records"),
    }


@app.get("/datasets")
def list_datasets(store: DatasetStore = Depends(get_store)) -> list[dict]:
    return [{k: v for k, v in m.items() if k != "training"} | {"status": m["training"]["status"]}
            for m in store.list()]


@app.get("/datasets/{dataset_id}")
def dataset_info(dataset_id: str, store: DatasetStore = Depends(get_store)) -> dict:
    return _meta_or_404(store, dataset_id)


@app.get("/datasets/{dataset_id}/data")
def dataset_data(
    dataset_id: str,
    kind: str = Query("original", pattern="^(original|synthetic)$"),
    store: DatasetStore = Depends(get_store),
):
    """Restituisce in CSV il dataset originale o l'ultimo sintetico generato."""
    _meta_or_404(store, dataset_id)
    f = store.path(dataset_id) / f"{kind}.csv"
    if not f.exists():
        raise HTTPException(404, "Nessun dataset sintetico generato.")
    return StreamingResponse(iter([f.read_text()]), media_type="text/csv")


@app.post("/train", status_code=202)
def train(req: TrainRequest, background: BackgroundTasks, store: DatasetStore = Depends(get_store)) -> dict:
    """Avvia l'addestramento del VAE in background."""
    meta = _meta_or_404(store, req.dataset_id)
    if meta["training"].get("status") == "running":
        raise HTTPException(409, "Addestramento gia' in corso per questo dataset.")
    unknown = set(req.categorical_columns or []) - set(meta["columns"])
    if unknown:
        raise HTTPException(400, f"Colonne inesistenti: {sorted(unknown)}")

    _models.pop(req.dataset_id, None)
    store.update_meta(req.dataset_id, training={"status": "running", "epoch": 0, "epochs": req.epochs, "history": []})
    background.add_task(_run_training, store, req)
    return {"dataset_id": req.dataset_id, "status": "running"}


@app.get("/train/{dataset_id}/status")
def train_status(dataset_id: str, store: DatasetStore = Depends(get_store)) -> dict:
    return _meta_or_404(store, dataset_id)["training"]


@app.get("/generate")
def generate(
    dataset_id: str,
    n_samples: int = Query(1000, ge=1, le=100_000),
    seed: int | None = None,
    store: DatasetStore = Depends(get_store),
):
    """Genera un dataset sintetico e lo restituisce come file CSV scaricabile."""
    _meta_or_404(store, dataset_id)
    model = _get_model(store, dataset_id)
    synth = model.generate(n_samples, seed=seed)
    store.save_synthetic(dataset_id, synth)

    buf = io.StringIO()
    synth.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="synthetic_{dataset_id}.csv"'},
    )


@app.post("/validate")
def validate(req: ValidateRequest, store: DatasetStore = Depends(get_store)) -> dict:
    """Metriche di qualita' (correlazione, distribuzioni) e privacy check."""
    _meta_or_404(store, req.dataset_id)
    try:
        synth = store.load_synthetic(req.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    real = store.load_original(req.dataset_id)
    model = _get_model(store, req.dataset_id)
    report = full_report(real, synth, model.preprocessor)
    table = comparison_table(real, synth)
    table.columns = [f"{a} ({b})" for a, b in table.columns]
    report["comparison_table"] = table.reset_index(names="column").to_dict("records")
    return report


@app.post("/ask")
def ask(req: AskRequest, store: DatasetStore = Depends(get_store)) -> dict:
    """Domanda in linguaggio naturale sul dataset sintetico."""
    meta = _meta_or_404(store, req.dataset_id)
    try:
        synth = store.load_synthetic(req.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    try:
        analyst = get_analyst()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    answer = analyst.ask(
        req.question,
        synth,
        schema=meta.get("schema"),
        history=[m.model_dump() for m in req.history],
    )
    return answer.to_dict()


def get_analyst() -> NaturalLanguageAnalyst:
    return NaturalLanguageAnalyst()
