"""Natural Language Analyst: l'utente fa domande in linguaggio naturale sul
dataset sintetico, l'LLM (OpenAI) genera codice Pandas + Matplotlib, il
backend lo esegue in modo sicuro e restituisce testo e grafici.

Se il codice generato va in errore, l'errore viene rimandato al modello
che ha un tentativo per correggersi.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd

from .config import OPENAI_MODEL
from .safe_exec import ExecutionResult, execute_code

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT_TEMPLATE = """Sei un data analyst esperto in Python. Lavori su un dataset \
tabellare SINTETICO, generato da un Variational Autoencoder a partire da dati reali sensibili.

Il dataset e' gia' caricato in un DataFrame pandas chiamato `df` ({n_rows} righe, {n_cols} colonne).

## Struttura del dataset
{schema}

## Statistiche descrittive
{stats}

## Prime righe
{head}

## Regole per il codice
- Rispondi con una breve spiegazione in italiano e UN SOLO blocco ```python```.
- Usa esclusivamente `df`, pandas (`pd`), numpy (`np`), matplotlib (`plt`), seaborn (`sns`) e scipy.
- NON leggere o scrivere file, NON usare rete, os, sys, subprocess, eval/exec, cicli while.
- NON chiamare plt.show() ne' plt.savefig(): i grafici vengono catturati automaticamente.
- Se la domanda richiede un grafico, crealo con matplotlib/seaborn, con titolo e assi etichettati.
- Salva la risposta numerica o tabellare principale nella variabile `result`
  (numero, stringa, Series o DataFrame) e stampa con print eventuali commenti sui risultati.
- Usa solo le colonne elencate sopra, rispettando maiuscole e minuscole.
- Se la domanda non riguarda il dataset, spiegalo senza generare codice.
"""

FIX_PROMPT = """Il codice che hai scritto ha generato questo errore:

{error}

Correggi il codice rispettando le stesse regole e restituisci di nuovo spiegazione + blocco ```python```."""


@dataclass
class AnalystAnswer:
    question: str
    explanation: str
    code: str | None
    execution: ExecutionResult | None
    attempts: int = 1
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "explanation": self.explanation,
            "code": self.code,
            "execution": self.execution.to_dict() if self.execution else None,
            "attempts": self.attempts,
        }


def describe_schema(df: pd.DataFrame, schema: list[dict] | None = None) -> str:
    lines = []
    if schema:
        for col in schema:
            extra = f" - valori: {', '.join(col['categories'])}" if col["categories"] else ""
            lines.append(f"- `{col['name']}`: {col['kind']} ({col['dtype']}){extra}")
    else:
        for name, dtype in df.dtypes.items():
            lines.append(f"- `{name}`: {dtype}")
    return "\n".join(lines)


def build_system_prompt(df: pd.DataFrame, schema: list[dict] | None = None) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        n_rows=len(df),
        n_cols=df.shape[1],
        schema=describe_schema(df, schema),
        stats=df.describe(include="all").round(2).to_string(),
        head=df.head(5).to_string(index=False),
    )


def extract_code(text: str) -> tuple[str, str | None]:
    """Separa la spiegazione testuale dal blocco di codice."""
    match = CODE_BLOCK_RE.search(text)
    if not match:
        return text.strip(), None
    code = match.group(1).strip()
    explanation = (text[: match.start()] + text[match.end():]).strip()
    return explanation, code


class NaturalLanguageAnalyst:
    def __init__(self, model: str | None = None, api_key: str | None = None, client=None):
        self.model = model or OPENAI_MODEL
        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY non impostata: aggiungila al file .env per usare il chatbot."
                )
            from openai import OpenAI

            self.client = OpenAI(api_key=key)

    def _chat(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def ask(
        self,
        question: str,
        df: pd.DataFrame,
        schema: list[dict] | None = None,
        history: list[dict] | None = None,
        max_retries: int = 1,
    ) -> AnalystAnswer:
        messages = [{"role": "system", "content": build_system_prompt(df, schema)}]
        # Mantengo solo gli ultimi scambi per non far crescere troppo il contesto
        messages += (history or [])[-6:]
        messages.append({"role": "user", "content": question})

        reply = self._chat(messages)
        explanation, code = extract_code(reply)
        if code is None:
            return AnalystAnswer(question, explanation, None, None, messages=messages)

        execution = execute_code(code, df)
        attempts = 1
        while not execution.success and attempts <= max_retries:
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": FIX_PROMPT.format(error=execution.error)},
            ]
            reply = self._chat(messages)
            new_expl, new_code = extract_code(reply)
            if new_code is None:
                break
            explanation, code = new_expl or explanation, new_code
            execution = execute_code(code, df)
            attempts += 1

        return AnalystAnswer(question, explanation, code, execution, attempts, messages)
