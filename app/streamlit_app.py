"""User Interface: dashboard Streamlit della Synthetic Data Sandbox.

La UI comunica con il backend FastAPI (SANDBOX_API_URL, default
http://localhost:8000).

Avvio:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests
import streamlit as st

from sandbox.config import API_URL
from sandbox.validation import plot_correlation_heatmaps, plot_distributions

st.set_page_config(page_title="Synthetic Data Sandbox", page_icon="🧬", layout="wide")


# ------------------------------------------------------------------ client API


def api(method: str, path: str, **kwargs) -> requests.Response:
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 120), **kwargs)
    except requests.ConnectionError:
        st.error(f"Backend non raggiungibile su {API_URL}. Avvia l'API con `uvicorn api.main:app`.")
        st.stop()
    return resp


def show_error(resp: requests.Response) -> None:
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    st.error(f"Errore {resp.status_code}: {detail}")


def loss_charts(history: list[dict], container=None) -> None:
    """Loss totale e ricostruzione in un grafico, KL in un altro: hanno scale molto diverse."""
    hist = pd.DataFrame(history).set_index("epoch")
    target = container.container() if container is not None else st.container()
    c1, c2 = target.columns(2)
    c1.markdown("**Loss totale e Reconstruction (MSE)**")
    c1.line_chart(hist[["loss", "recon_loss"]])
    c2.markdown("**KL Divergence**")
    c2.line_chart(hist[["kl_loss"]])



@st.cache_data(show_spinner=False)
def fetch_csv(dataset_id: str, kind: str, version: str) -> pd.DataFrame:
    """``version`` serve solo a invalidare la cache dopo una nuova generazione."""
    resp = api("GET", f"/datasets/{dataset_id}/data", params={"kind": kind})
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


# --------------------------------------------------------------------- stato

ss = st.session_state
ss.setdefault("dataset_id", None)
ss.setdefault("synth_version", "0")
ss.setdefault("chat", [])
ss.setdefault("report", None)


# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🧬 Synthetic Data Sandbox")
    st.caption("VAE Edition")

    health = api("GET", "/health", timeout=5)
    if health.ok:
        st.success(f"API online · v{health.json()['version']}")
    else:
        st.error("API non disponibile")

    datasets = api("GET", "/datasets").json()
    if datasets:
        labels = {d["dataset_id"]: f"{d['filename']} · {d['dataset_id']} ({d['status']})" for d in datasets}
        ids = list(labels)
        current = ids.index(ss.dataset_id) if ss.dataset_id in ids else 0
        chosen = st.selectbox("Dataset", ids, index=current, format_func=labels.get)
        if chosen != ss.dataset_id:
            ss.dataset_id = chosen
            ss.chat, ss.report = [], None
    else:
        st.info("Nessun dataset caricato.")

    st.divider()
    st.markdown(
        "**Flusso di lavoro**\n"
        "1. Carica un CSV\n"
        "2. Addestra il VAE\n"
        "3. Genera e valida i dati sintetici\n"
        "4. Interroga i dati con il chatbot"
    )

tab_upload, tab_train, tab_generate, tab_chat = st.tabs(
    ["📤 Upload", "🧠 Training", "🧪 Generazione e validazione", "💬 Chatbot"]
)


# -------------------------------------------------------------------- upload

with tab_upload:
    st.header("Caricamento dataset")
    st.write(
        "Carica un dataset tabellare *sensibile* (es. dati medici o finanziari). "
        "Le variabili numeriche vengono normalizzate con MinMaxScaler, le categoriche "
        "codificate con One-Hot Encoding e i valori nulli imputati."
    )
    uploaded = st.file_uploader("Trascina qui il file CSV", type=["csv"])
    if uploaded is not None and st.button("Carica dataset", type="primary"):
        resp = api("POST", "/upload", files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")})
        if resp.ok:
            info = resp.json()
            ss.dataset_id = info["dataset_id"]
            ss.chat, ss.report = [], None
            st.success(f"Dataset caricato con id `{info['dataset_id']}`")
            st.rerun()
        else:
            show_error(resp)

    if ss.dataset_id:
        real = fetch_csv(ss.dataset_id, "original", "orig")
        st.subheader(f"Anteprima · {real.shape[0]} righe × {real.shape[1]} colonne")
        st.dataframe(real.head(50), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tipi e valori nulli**")
            st.dataframe(
                pd.DataFrame({"tipo": real.dtypes.astype(str), "nulli": real.isna().sum(),
                              "% nulli": (real.isna().mean() * 100).round(1)}),
                width="stretch",
            )
        with c2:
            st.markdown("**Statistiche descrittive**")
            st.dataframe(real.describe().T.round(2), width="stretch")


# ------------------------------------------------------------------ training

with tab_train:
    st.header("Addestramento del VAE")
    if not ss.dataset_id:
        st.info("Carica prima un dataset.")
    else:
        meta = api("GET", f"/datasets/{ss.dataset_id}").json()
        with st.form("train_form"):
            c1, c2, c3 = st.columns(3)
            epochs = c1.number_input("Epoche", 10, 5000, 300, step=50)
            batch_size = c1.number_input("Batch size", 8, 1024, 64, step=8)
            latent_dim = c2.number_input("Dimensione spazio latente", 1, 128, 8)
            hidden = c2.text_input("Strati nascosti (separati da virgola)", "64, 32")
            lr = c3.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.5f")
            beta = c3.number_input("Beta (peso KL)", 0.0, 10.0, 0.01, step=0.005, format="%.3f")
            cat_cols = st.multiselect(
                "Colonne da trattare come categoriche (oltre a quelle rilevate automaticamente)",
                meta["columns"],
            )
            submitted = st.form_submit_button("Avvia addestramento", type="primary")

        if submitted:
            try:
                hidden_dims = [int(x) for x in hidden.split(",") if x.strip()]
            except ValueError:
                st.error("Formato strati nascosti non valido.")
                st.stop()
            resp = api("POST", "/train", json={
                "dataset_id": ss.dataset_id, "epochs": int(epochs), "batch_size": int(batch_size),
                "latent_dim": int(latent_dim), "hidden_dims": hidden_dims, "learning_rate": float(lr),
                "beta": float(beta), "categorical_columns": cat_cols or None,
            })
            if not resp.ok:
                show_error(resp)
            else:
                bar = st.progress(0.0, text="Addestramento in corso...")
                chart = st.empty()
                while True:
                    status = api("GET", f"/train/{ss.dataset_id}/status").json()
                    if status.get("history"):
                        loss_charts(status["history"], chart)
                    if status["status"] == "running":
                        bar.progress(status.get("epoch", 0) / status.get("epochs", 1),
                                     text=f"Epoca {status.get('epoch', 0)}/{status.get('epochs')}")
                        time.sleep(1)
                        continue
                    break
                if status["status"] == "completed":
                    bar.progress(1.0, text="Addestramento completato")
                    st.success("Modello addestrato. Passa alla generazione dei dati sintetici.")
                    ss.report = None
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Addestramento fallito: {status.get('error')}")

        status = api("GET", f"/train/{ss.dataset_id}/status").json()
        if status["status"] == "completed" and not submitted:
            st.success("Modello gia' addestrato per questo dataset.")
            st.markdown("Andamento della loss: `MSE + β · KL`")
            loss_charts(status["history"])
            st.json(status.get("config", {}), expanded=False)


# ---------------------------------------------------- generazione/validazione

with tab_generate:
    st.header("Generazione e validazione")
    if not ss.dataset_id:
        st.info("Carica prima un dataset.")
    else:
        status = api("GET", f"/train/{ss.dataset_id}/status").json()
        if status["status"] != "completed":
            st.info("Addestra prima il modello.")
        else:
            real = fetch_csv(ss.dataset_id, "original", "orig")
            c1, c2, c3 = st.columns([2, 2, 1])
            n_samples = c1.number_input("Numero di record sintetici", 10, 100_000, len(real), step=100)
            seed = c2.number_input("Seed (opzionale, -1 = casuale)", -1, 10**6, -1)
            c3.write("")
            c3.write("")
            if c3.button("Genera", type="primary"):
                params = {"dataset_id": ss.dataset_id, "n_samples": int(n_samples)}
                if seed >= 0:
                    params["seed"] = int(seed)
                resp = api("GET", "/generate", params=params)
                if not resp.ok:
                    show_error(resp)
                else:
                    ss.synth_version = str(time.time())
                    val = api("POST", "/validate", json={"dataset_id": ss.dataset_id})
                    ss.report = val.json() if val.ok else None

            try:
                synth = fetch_csv(ss.dataset_id, "synthetic", ss.synth_version)
            except requests.HTTPError:
                synth = None

            if synth is None:
                st.info("Nessun dataset sintetico ancora generato.")
            else:
                if ss.report is None:
                    val = api("POST", "/validate", json={"dataset_id": ss.dataset_id})
                    ss.report = val.json() if val.ok else None
                report = ss.report

                st.download_button(
                    "⬇️ Scarica dataset sintetico (CSV)",
                    synth.to_csv(index=False).encode(),
                    file_name=f"synthetic_{ss.dataset_id}.csv",
                    mime="text/csv",
                )

                if report:
                    priv = report["privacy"]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Fidelity score", f"{report['fidelity_score']:.1%}",
                              help="Media di (1 - KS) per le numeriche e (1 - TVD) per le categoriche")
                    m2.metric("Δ correlazione media", f"{report['correlation']['mean_abs_diff']:.3f}",
                              help="Media della differenza assoluta tra le matrici di correlazione")
                    m3.metric("Record identici ai reali", priv["exact_matches"])
                    m4.metric("Privacy check", "✅ Superato" if priv["passed"] else "⚠️ Da verificare")

                st.subheader("Dati reali vs sintetici")
                c1, c2 = st.columns(2)
                c1.markdown("**Reali**")
                c1.dataframe(real.head(15), width="stretch")
                c2.markdown("**Sintetici**")
                c2.dataframe(synth.head(15), width="stretch")

                if report:
                    st.markdown("**Tabella comparativa delle statistiche**")
                    st.dataframe(pd.DataFrame(report["comparison_table"]).set_index("column"),
                                 width="stretch")

                    st.markdown("**Similarita' delle distribuzioni per colonna**")
                    dist = pd.DataFrame(report["distribution"])
                    cols = [c for c in ["column", "type", "similarity", "ks_statistic", "ks_pvalue", "tvd",
                                        "real_mean", "synth_mean", "real_std", "synth_std"] if c in dist]
                    st.dataframe(dist[cols].set_index("column").round(3), width="stretch")

                st.subheader("Correlazioni")
                st.pyplot(plot_correlation_heatmaps(real, synth))

                st.subheader("Distribuzioni")
                selected = st.multiselect("Variabili", list(real.columns), default=list(real.columns))
                if selected:
                    st.pyplot(plot_distributions(real, synth, selected))

                if report:
                    st.subheader("Privacy check")
                    st.write(
                        "Verifico che il modello non abbia memorizzato i record reali: oltre alle copie "
                        "esatte calcolo la *Distance to Closest Record* (DCR) di ogni record sintetico dal "
                        "reale piu' vicino e la confronto con la distanza tipica tra i record reali."
                    )
                    st.dataframe(pd.Series(priv, name="valore").astype(str), width="stretch")


# ------------------------------------------------------------------- chatbot

with tab_chat:
    st.header("Natural Language Analyst")
    st.caption(
        "Fai una domanda sui dati sintetici (es. *\"Qual è la correlazione tra glucosio e BMI?\"*). "
        "L'LLM genera codice Pandas + Matplotlib che viene eseguito in sandbox sul backend."
    )

    if not ss.dataset_id:
        st.info("Carica prima un dataset.")
    else:
        def render_answer(ans: dict) -> None:
            if ans.get("explanation"):
                st.markdown(ans["explanation"])
            ex = ans.get("execution")
            if ex:
                if ex["success"]:
                    for img in ex["images"]:
                        st.image(base64.b64decode(img))
                    if ex.get("result"):
                        st.markdown(ex["result"])
                    if ex.get("stdout"):
                        st.code(ex["stdout"], language="text")
                else:
                    st.error(f"Errore nell'esecuzione del codice: {ex['error']}")
            if ans.get("code"):
                with st.expander("Codice generato"):
                    st.code(ans["code"], language="python")

        if st.button("🗑️ Svuota chat"):
            ss.chat = []

        for msg in ss.chat:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_answer(msg["answer"])

        question = st.chat_input("Scrivi una domanda sui dati sintetici...")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            history = []
            for msg in ss.chat:
                if msg["role"] == "user":
                    history.append({"role": "user", "content": msg["content"]})
                else:
                    a = msg["answer"]
                    text = a.get("explanation") or ""
                    if a.get("code"):
                        text += f"\n```python\n{a['code']}\n```"
                    history.append({"role": "assistant", "content": text})

            with st.chat_message("assistant"):
                with st.spinner("Sto analizzando i dati..."):
                    resp = api("POST", "/ask", json={
                        "dataset_id": ss.dataset_id, "question": question, "history": history,
                    }, timeout=180)
                if resp.ok:
                    answer = resp.json()
                    render_answer(answer)
                    ss.chat += [{"role": "user", "content": question},
                                {"role": "assistant", "answer": answer}]
                else:
                    show_error(resp)
