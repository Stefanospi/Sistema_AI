"""Test del server FastAPI: client simulato, nessuna chiamata di rete reale.

Copre:
- GET /v1/models: lista modelli virtuali corretta
- POST /v1/chat/completions modalità non-stream (proxy + auto-classify)
- POST /v1/chat/completions modalità stream (SSE, proxy + auto-classify)
- Gestione errori (body vuoto, messages mancanti)
- Autenticazione Bearer opzionale
- Fallback tra modelli anche via HTTP
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import del server PRIMA di importare router (il server usa import lazy)
import server  # noqa: E402  # type: ignore

from conftest import FakeClient, conn_error, msg  # noqa: E402

# Evita che il server tenti di caricare il Router reale durante i test
from fastapi.testclient import TestClient  # noqa: E402

M1 = "nvidia/nemotron-3.5-super-vl:free"
M2 = "thinkingmachines/inkling-small:free"
GLOBAL = "openrouter/free"


@pytest.fixture(autouse=True)
def reset_router_singleton():
    """Resetta il singleton del router prima di ogni test."""
    server._router_instance = None
    yield
    server._router_instance = None


@pytest.fixture
def fake_router(tmp_path, monkeypatch):
    """
    Inietta un Router con FakeClient nel singleton del server.
    Restituisce (client_http, fake_openai_client).
    """
    from router import Memory, Router

    def make(script: dict):
        fc = FakeClient(script)
        r = Router(
            client=fc,
            memory=Memory(tmp_path / "m.db"),
            log_file=tmp_path / "test.log",
        )
        server._router_instance = r
        return TestClient(server.app, raise_server_exceptions=False), fc

    return make


# ---------- GET /v1/models ----------

def test_list_models_contiene_auto_e_categorie(fake_router):
    client, _ = fake_router({})
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = [m["id"] for m in data["data"]]
    assert "router-auto" in ids
    assert "router-coding" in ids
    assert "router-finanza" in ids
    assert "router-appunti" in ids
    assert "router-generale" in ids
    assert all(m["object"] == "model" for m in data["data"])


# ---------- POST non-stream: modalità proxy ----------

def test_proxy_non_stream_router_auto(fake_router):
    """router-auto classifica automaticamente e restituisce una risposta valida."""
    client, fc = fake_router({M1: [msg("Risposta codice")], M2: [msg("alt")]})
    resp = client.post("/v1/chat/completions", json={
        "model": "router-auto",
        "messages": [{"role": "user", "content": "c'è un bug nella funzione python"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Risposta codice"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert fc.calls[0]["model"] == M1


def test_proxy_non_stream_categoria_esplicita(fake_router):
    """router-appunti usa direttamente i modelli della categoria appunti."""
    LIGHT = "nvidia/nemotron-3.5-lightning:free"
    client, fc = fake_router({LIGHT: [msg("Appunto registrato")]})
    resp = client.post("/v1/chat/completions", json={
        "model": "router-appunti",
        "messages": [{"role": "user", "content": "segna che devo chiamare Mario"}],
    })
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Appunto registrato"
    assert fc.calls[0]["model"] == LIGHT


def test_proxy_non_stream_fallback(fake_router):
    """Se il primo modello fallisce, il server prova il successivo."""
    client, fc = fake_router({M1: [conn_error()], M2: [msg("ok dal fallback")]})
    resp = client.post("/v1/chat/completions", json={
        "model": "router-generale",
        "messages": [{"role": "user", "content": "ciao"}],
    })
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok dal fallback"
    assert [c["model"] for c in fc.calls] == [M1, M2]


def test_proxy_non_stream_tutti_falliscono(fake_router):
    """Se tutti i modelli falliscono, risponde 502 con formato errore OpenAI."""
    client, fc = fake_router({M1: [conn_error()], M2: [conn_error()], GLOBAL: [conn_error()]})
    resp = client.post("/v1/chat/completions", json={
        "model": "router-generale",
        "messages": [{"role": "user", "content": "ciao"}],
    })
    assert resp.status_code == 502
    body = resp.json()
    assert "error" in body
    assert "Tutti i modelli" in body["error"]["message"]


def test_proxy_messages_mancanti(fake_router):
    client, _ = fake_router({})
    resp = client.post("/v1/chat/completions", json={"model": "router-auto"})
    assert resp.status_code == 400
    assert "messages" in resp.json()["error"]["message"]


def test_proxy_body_non_json(fake_router):
    client, _ = fake_router({})
    resp = client.post("/v1/chat/completions", content=b"non json!", headers={"content-type": "application/json"})
    assert resp.status_code == 400


def test_proxy_inoltra_tools(fake_router):
    """I tools definiti dall'editor vengono passati al modello reale."""
    tools = [{"type": "function", "function": {"name": "search", "description": "cerca", "parameters": {}}}]
    client, fc = fake_router({M1: [msg("trovato")]})
    resp = client.post("/v1/chat/completions", json={
        "model": "router-coding",
        "messages": [{"role": "user", "content": "cerca qualcosa"}],
        "tools": tools,
    })
    assert resp.status_code == 200
    # Verifica che i tools siano stati passati al FakeClient
    assert fc.calls[0].get("tools") == tools


# ---------- POST stream: modalità proxy ----------

def _parse_sse(text: str) -> list[dict]:
    """Analizza l'output SSE e restituisce i chunk JSON (esclude [DONE])."""
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:") and "[DONE]" not in line:
            chunks.append(json.loads(line[5:].strip()))
    return chunks


def test_proxy_stream_emette_sse(fake_router):
    client, fc = fake_router({M1: [msg("ciao mondo")]})
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "router-auto",
        "messages": [{"role": "user", "content": "ciao bello"}],
        "stream": True,
    }) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        raw = resp.read().decode("utf-8")

    chunks = _parse_sse(raw)
    # Almeno un chunk di testo + un chunk finale con finish_reason=stop
    text_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
    stop_chunks = [c for c in chunks if c["choices"][0].get("finish_reason") == "stop"]
    assert text_chunks, "Nessun chunk di testo ricevuto"
    assert stop_chunks, "Nessun chunk finale con finish_reason=stop"
    full_text = "".join(c["choices"][0]["delta"]["content"] for c in text_chunks)
    assert full_text == "ciao mondo"


def test_proxy_stream_fallback_prima_del_testo(fake_router):
    """Se il primo modello fallisce prima di emettere testo, il fallback avviene trasparentemente."""
    client, fc = fake_router({M1: [conn_error()], M2: [msg("ok")]})
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "router-generale",
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }) as resp:
        raw = resp.read().decode("utf-8")

    chunks = _parse_sse(raw)
    text_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
    assert "".join(c["choices"][0]["delta"]["content"] for c in text_chunks) == "ok"
    # Il secondo modello è stato usato
    assert any(c["model"] == M2 for c in chunks)


def test_proxy_stream_router_auto_classifica(fake_router):
    """Con stream=True e router-auto, la classificazione avviene correttamente."""
    LIGHT = "thinkingmachines/inkling-small:free"  # primo modello della cat. finanza in config
    client, fc = fake_router({LIGHT: [msg("Risposta finanza")]})
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "router-auto",
        "messages": [{"role": "user", "content": "quanto rende un ETF?"}],
        "stream": True,
    }) as resp:
        raw = resp.read().decode("utf-8")

    chunks = _parse_sse(raw)
    text_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
    assert "".join(c["choices"][0]["delta"]["content"] for c in text_chunks) == "Risposta finanza"


# ---------- autenticazione Bearer ----------

def test_auth_token_accettato(fake_router, monkeypatch):
    monkeypatch.setenv("ROUTER_API_TOKEN", "secret-token-test")
    client, _ = fake_router({})
    resp = client.get("/v1/models", headers={"Authorization": "Bearer secret-token-test"})
    assert resp.status_code == 200


def test_auth_token_rifiutato(fake_router, monkeypatch):
    monkeypatch.setenv("ROUTER_API_TOKEN", "secret-token-test")
    client, _ = fake_router({})
    resp = client.get("/v1/models", headers={"Authorization": "Bearer sbagliato"})
    assert resp.status_code == 401


def test_auth_senza_token_configurato_libero(fake_router, monkeypatch):
    monkeypatch.delenv("ROUTER_API_TOKEN", raising=False)
    client, _ = fake_router({})
    resp = client.get("/v1/models")
    assert resp.status_code == 200
