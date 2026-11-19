"""Esecuzione "sicura" del codice Python generato dall'LLM.

Il codice prodotto dal modello non e' affidabile, quindi uso piu' livelli
di protezione:

1. analisi statica con ``ast``: solo import da una whitelist, niente
   attributi "dunder"/privati, niente funzioni pericolose (eval, exec,
   open...) e niente metodi pandas che leggono/scrivono file;
2. esecuzione con ``__builtins__`` ridotti e un ``__import__`` filtrato;
3. esecuzione in un processo separato con timeout e limite di CPU,
   cosi' un loop infinito o un crash non bloccano il backend.

Il codice lavora su una copia del DataFrame ``df`` e puo' restituire:
- un grafico matplotlib (variabile ``fig`` o figura corrente);
- un valore testuale/tabellare nella variabile ``result``;
- output stampato con ``print``.
"""

from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import io
import multiprocessing as mp
import traceback
from dataclasses import dataclass, field

import pandas as pd

ALLOWED_MODULES = {
    "pandas", "numpy", "matplotlib", "seaborn", "scipy", "math", "statistics",
}

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit",
    "help", "memoryview", "classmethod", "staticmethod", "super", "type",
    "object", "dir",
}

# Metodi che toccano il filesystem, la rete o eseguono stringhe di codice
FORBIDDEN_ATTRS = {
    "eval", "system", "popen", "savefig", "load", "save", "loadtxt", "savetxt",
    "fromfile", "tofile", "memmap", "imread", "imsave",
    "read_csv", "read_excel", "read_json", "read_html", "read_sql", "read_pickle",
    "read_parquet", "read_table", "read_clipboard", "read_feather", "read_hdf",
    "read_xml", "read_fwf", "read_orc", "read_sas", "read_spss", "read_stata",
    "to_csv", "to_excel", "to_json", "to_html", "to_sql", "to_pickle",
    "to_parquet", "to_clipboard", "to_feather", "to_hdf", "to_xml", "to_latex",
    "to_stata", "to_orc", "pipe", "apply_along_axis",
}

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "format", "frozenset", "int", "isinstance", "len", "list", "map", "max",
        "min", "print", "range", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "zip", "True", "False", "None", "ValueError",
        "KeyError", "TypeError", "Exception", "ZeroDivisionError", "IndexError",
    ]
}

DEFAULT_TIMEOUT = 20  # secondi


class UnsafeCodeError(ValueError):
    """Il codice non ha superato la validazione statica."""


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    result: str | None = None
    images: list[str] = field(default_factory=list)  # PNG in base64
    error: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------- validazione AST


def validate_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Errore di sintassi: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] not in ALLOWED_MODULES:
                    raise UnsafeCodeError(f"Import non consentito: {name}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCodeError(f"Accesso ad attributo privato non consentito: {node.attr}")
            if node.attr in FORBIDDEN_ATTRS:
                raise UnsafeCodeError(f"Metodo non consentito: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise UnsafeCodeError(f"Funzione non consentita: {node.id}")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise UnsafeCodeError("Costrutto non consentito.")
        elif isinstance(node, (ast.While,)):
            # i while sono la causa piu' comune di loop infiniti; per l'analisi
            # dati bastano for e operazioni vettoriali
            raise UnsafeCodeError("Il ciclo while non e' consentito.")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"Import non consentito: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


# ----------------------------------------------------------- esecuzione


def _format_result(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (pd.DataFrame, pd.Series)):
        try:
            return value.to_markdown()
        except ImportError:
            return value.to_string()
    return str(value)


def _run(code: str, df: pd.DataFrame, queue) -> None:
    """Eseguito nel processo figlio."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (DEFAULT_TIMEOUT, DEFAULT_TIMEOUT + 5))
    except (ImportError, ValueError, OSError):
        pass

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import
    env = {
        "__builtins__": safe_builtins,
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
    }

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<llm_code>", "exec"), env)  # noqa: S102

        images = []
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            images.append(base64.b64encode(buf.getvalue()).decode())
        plt.close("all")

        queue.put(ExecutionResult(
            success=True,
            stdout=stdout.getvalue()[:10_000],
            result=_format_result(env.get("result")),
            images=images,
        ).to_dict())
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exception_only(type(exc), exc)
        queue.put(ExecutionResult(
            success=False, stdout=stdout.getvalue()[:10_000], error="".join(tb).strip()
        ).to_dict())


def execute_code(code: str, df: pd.DataFrame, timeout: int = DEFAULT_TIMEOUT) -> ExecutionResult:
    try:
        validate_code(code)
    except UnsafeCodeError as exc:
        return ExecutionResult(success=False, error=str(exc))

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run, args=(code, df.copy(), queue), daemon=True)
    proc.start()
    try:
        data = queue.get(timeout=timeout)
    except Exception:  # noqa: BLE001 - queue.Empty o processo morto
        data = None
    proc.join(timeout=1)
    if proc.is_alive():
        proc.kill()
        proc.join()

    if data is None:
        return ExecutionResult(success=False, error=f"Esecuzione interrotta: superato il timeout di {timeout}s.")
    return ExecutionResult(**data)
