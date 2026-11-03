"""Capri sul telefono: spento senza token, token e origine, blocco dopo i tentativi, solo /m da fuori."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import mobile  # noqa: E402
from backends import BackendError  # noqa: E402
from test_brain import StubBackend, make_brain, ok_events  # noqa: E402,F401
from test_screen import served  # noqa: E402,F401

TOKEN = "t" * 32
AUTH = {"x-capri-token": TOKEN}


@pytest.fixture
def phone(served, monkeypatch):
    """Telefono acceso: abilitato in config e token in ambiente."""
    monkeypatch.setenv("MOBILE_TOKEN", TOKEN)
    monkeypatch.delenv("ROUTER_API_TOKEN", raising=False)
    mobile.guard.fails.clear()

    def start(backend=None, enabled=True):
        client, brain, router = served(backend or StubBackend(ok_events("Ciao dal PC.")))
        router.cfg["mobile"] = {"abilitato": enabled}
        return client, brain, router

    return start


def test_spento_di_default_o_senza_token(phone, monkeypatch):
    client, _, _ = phone(enabled=False)
    assert client.get("/m").status_code == 404
    assert client.post("/m/ask", headers=AUTH, json={"text": "ciao"}).status_code == 404
    client, _, _ = phone()
    monkeypatch.setenv("MOBILE_TOKEN", "corto")
    assert client.get("/m").status_code == 404                      # token troppo corto: come se non ci fosse
    monkeypatch.setenv("ROUTER_API_TOKEN", TOKEN)
    monkeypatch.setenv("MOBILE_TOKEN", TOKEN)
    assert client.get("/m").status_code == 404                      # uguale al token del router: rifiutato


def test_pagina_e_risposta(phone):
    client, _, router = phone()
    page = client.get("/m")
    assert page.status_code == 200 and "CAPRI" in page.text and page.headers["cache-control"] == "no-store"
    r = client.post("/m/ask", headers=AUTH, json={"text": "ciao"})
    assert r.status_code == 200 and r.json() == {"text": "Ciao dal PC."}
    assert [m for _, m, _ in router.memory.active(mobile.SESSION)] == ["user", "assistant"]
    assert client.post("/m/reset", headers=AUTH).json() == {"cleared": 2}


def test_token_e_blocco(phone):
    client, _, _ = phone()
    assert client.post("/m/ask", json={"text": "ciao"}).status_code == 401
    for _ in range(mobile.MAX_FAILS - 1):
        assert client.post("/m/ask", headers={"x-capri-token": "no"}, json={"text": "ciao"}).status_code == 401
    # a questo punto la porta è chiusa, anche con il token giusto
    assert client.post("/m/ask", headers=AUTH, json={"text": "ciao"}).status_code == 429


def test_origine_e_validazione(phone):
    client, _, _ = phone()
    ok = client.post("/m/ask", headers={**AUTH, "origin": "http://testserver"}, json={"text": "ciao"})
    assert ok.status_code == 200                                    # stessa origine della pagina
    bad = client.post("/m/ask", headers={**AUTH, "origin": "https://evil.example"}, json={"text": "ciao"})
    assert bad.status_code == 403
    assert client.post("/m/ask", headers=AUTH, json={"text": ""}).status_code == 422
    assert client.post("/m/ask", headers=AUTH, json={"text": "x" * 4001}).status_code == 422


def test_errore_del_cervello_arriva_leggibile(phone):
    client, _, _ = phone(StubBackend(error=BackendError("quota finita")))
    r = client.post("/m/ask", headers=AUTH, json={"text": "ciao"})
    assert r.status_code == 502 and "quota finita" in r.json()["error"]


def test_da_fuori_si_vede_solo_m(phone):
    client, _, _ = phone()
    tunnel = {"x-forwarded-for": "100.64.0.9", "tailscale-user-login": "io@example.com"}
    assert client.get("/health", headers=tunnel).status_code == 403
    assert client.post("/v1/chat/completions", headers=tunnel, json={}).status_code == 403
    assert client.get("/m", headers=tunnel).status_code == 200
    assert client.post("/m/ask", headers={**tunnel, **AUTH}, json={"text": "ciao"}).status_code == 200
    assert client.get("/health").status_code == 200                 # dal PC resta tutto com'era
