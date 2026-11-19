import io
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as main
from sandbox.storage import DatasetStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_store", DatasetStore(tmp_path))
    main._models.clear()
    return TestClient(main.app)


def _upload(client, df):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    resp = client.post("/upload", files={"file": ("dati.csv", buf, "text/csv")})
    assert resp.status_code == 200, resp.text
    return resp.json()["dataset_id"]


def _wait_training(client, dataset_id, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        status = client.get(f"/train/{dataset_id}/status").json()
        if status["status"] != "running":
            return status
        time.sleep(0.2)
    raise TimeoutError


def test_full_flow(client, mixed_df, monkeypatch):
    dataset_id = _upload(client, mixed_df)

    resp = client.get(f"/generate?dataset_id={dataset_id}")
    assert resp.status_code == 409  # modello non ancora addestrato

    resp = client.post("/train", json={"dataset_id": dataset_id, "epochs": 5})
    assert resp.status_code == 202
    status = _wait_training(client, dataset_id)
    assert status["status"] == "completed"
    assert len(status["history"]) == 5

    resp = client.get(f"/generate?dataset_id={dataset_id}&n_samples=40&seed=3")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert len(resp.text.strip().splitlines()) == 41

    report = client.post("/validate", json={"dataset_id": dataset_id}).json()
    assert {"fidelity_score", "correlation", "distribution", "privacy", "comparison_table"} <= set(report)

    reply = "```python\nresult = len(df)\n```"
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    )))
    monkeypatch.setattr(main, "get_analyst", lambda: main.NaturalLanguageAnalyst(client=fake))
    ans = client.post("/ask", json={"dataset_id": dataset_id, "question": "Quante righe?"}).json()
    assert ans["execution"]["result"] == "40"


def test_upload_rejects_non_csv(client):
    resp = client.post("/upload", files={"file": ("dati.txt", b"a,b\n1,2", "text/plain")})
    assert resp.status_code == 400


def test_unknown_dataset(client):
    assert client.get("/datasets/../../etc").status_code == 404
    assert client.get("/datasets/abcdefabcdef").status_code == 404
