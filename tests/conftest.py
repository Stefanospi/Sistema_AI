"""Utilità di test: client OpenAI simulato, nessuna chiamata di rete."""
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, BadRequestError, NotFoundError, RateLimitError

sys.path.insert(0, str(Path(__file__).parent.parent))

from router import Memory, Router  # noqa: E402


def msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def tool_call(name, arguments, id="call_1"):
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=arguments))


def rate_limit(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://x"))
    return RateLimitError("429", response=resp, body=None)


def conn_error():
    return APIConnectionError(request=httpx.Request("POST", "http://x"))


def status_error(code):
    resp = httpx.Response(code, request=httpx.Request("POST", "http://x"))
    cls = {400: BadRequestError, 404: NotFoundError}[code]
    return cls(str(code), response=resp, body=None)


class StreamBreak:
    """Stream che emette alcuni pezzi di testo e poi solleva un errore."""

    def __init__(self, pieces, exc):
        self.pieces, self.exc = pieces, exc


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _to_chunks(m):
    """Trasforma un messaggio completo in chunk di streaming (testo a pezzi, tool call spezzate)."""
    if m.content:
        half = max(len(m.content) // 2, 1)
        yield _chunk(m.content[:half])
        yield _chunk(m.content[half:])
    for i, c in enumerate(m.tool_calls or []):
        a = c.function.arguments
        yield _chunk(tool_calls=[SimpleNamespace(index=i, id=c.id, function=SimpleNamespace(
            name=c.function.name, arguments=a[:3]))])
        yield _chunk(tool_calls=[SimpleNamespace(index=i, id=None, function=SimpleNamespace(
            name=None, arguments=a[3:]))])


class FakeClient:
    """script: dict modello -> lista di esiti (messaggio o eccezione), consumati in ordine."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages], **kwargs})
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if kwargs.get("stream"):
            if isinstance(outcome, StreamBreak):
                def broken():
                    for p in outcome.pieces:
                        yield _chunk(p)
                    raise outcome.exc
                return broken()
            if outcome == "no_choices":
                return iter([SimpleNamespace(choices=[])])
            return _to_chunks(outcome)
        if outcome == "no_choices":
            return SimpleNamespace(choices=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=outcome)])


@pytest.fixture
def make_router(tmp_path):
    sleeps = []

    def factory(script, **cfg_overrides):
        r = Router(client=FakeClient(script), memory=Memory(tmp_path / "m.db"), sleep=sleeps.append,
                   log_file=tmp_path / "test.log")
        r.cfg.update(cfg_overrides)
        return r

    factory.sleeps = sleeps
    factory.log_path = tmp_path / "test.log"
    return factory
