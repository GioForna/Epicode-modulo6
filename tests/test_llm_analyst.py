from types import SimpleNamespace

import pandas as pd

from sandbox.llm_analyst import NaturalLanguageAnalyst, build_system_prompt, extract_code


class FakeClient:
    """Client finto che restituisce le risposte in ordine."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


DF = pd.DataFrame({"reddito": [10, 20, 30, 40], "risparmio": [1, 2, 3, 5]})


def test_extract_code():
    text = "Ecco l'analisi.\n```python\nresult = 1\n```\nFine."
    expl, code = extract_code(text)
    assert code == "result = 1"
    assert "Ecco" in expl and "Fine" in expl
    assert extract_code("solo testo") == ("solo testo", None)


def test_system_prompt_contains_schema():
    prompt = build_system_prompt(DF)
    assert "reddito" in prompt and "risparmio" in prompt and "`df`" in prompt


def test_ask_executes_code():
    client = FakeClient(["Calcolo la correlazione.\n```python\nresult = round(df['reddito'].corr(df['risparmio']), 2)\n```"])
    ans = NaturalLanguageAnalyst(client=client).ask("Correlazione tra reddito e risparmio?", DF)
    assert ans.execution.success
    assert float(ans.execution.result) > 0.9
    assert client.calls[0]["messages"][0]["role"] == "system"


def test_ask_self_correction():
    client = FakeClient([
        "```python\nresult = df['Reddito'].mean()\n```",
        "Corretto.\n```python\nresult = df['reddito'].mean()\n```",
    ])
    ans = NaturalLanguageAnalyst(client=client).ask("Media reddito?", DF)
    assert ans.attempts == 2
    assert ans.execution.success and float(ans.execution.result) == 25.0
