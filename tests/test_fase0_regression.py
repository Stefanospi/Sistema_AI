"""Regressioni Fase 0.

Bug (a): server.py _proxy_complete - cache salvava response.body (bytes),
         al cache hit JSONResponse(content=cached) sollevava TypeError.
         Fix: la cache ora contiene il dict Python.

Bug (b): tools.py Toolbox.execute - try/return era DENTRO il for loop di
         validazione tipi, quindi:
           1. Con args vuoti (clean.items() vuoto) restituiva None.
           2. Con un solo argomento eseguiva il tool prima di validare gli altri.
         Fix: loop separato di validazione, poi chiamata unica.

Bug (c): memoria.db-shm e memoria.db-wal mancavano da .gitignore.
         Verificato tramite presenza nel file.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import server  # noqa: E402
from cache import TTLCache  # noqa: E402
from conftest import FakeClient, msg  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from tools import Toolbox  # noqa: E402

M1 = "nvidia/nemotron-3.5-super-vl:free"


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_router_singleton():
    server._router_instance = None
    yield
    server._router_instance = None


@pytest.fixture
def fake_router(tmp_path, monkeypatch):
    from router import Memory, Router

    def make(script):
        fc = FakeClient(script)
        r = Router(
            client=fc,
            memory=Memory(tmp_path / "m.db"),
            log_file=tmp_path / "test.log",
        )
        server._router_instance = r
        return TestClient(server.app, raise_server_exceptions=False), fc

    return make


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _tb(ws, answer=True, **cfg):
    return Toolbox(ws, cfg, confirm=lambda _: answer)


# ─── Bug (a): cache dict vs bytes ─────────────────────────────────────────────

class TestCacheDictBug:
    """La seconda richiesta identica deve restituire una risposta valida (non TypeError)."""

    def test_cache_hit_restituisce_json_valido(self, fake_router):
        """Due richieste identiche: la seconda viene servita dalla cache senza errori."""
        client, fc = fake_router({M1: [msg("risposta cached")]})

        payload = {
            "model": "router-generale",
            "messages": [{"role": "user", "content": "prova cache"}],
        }
        # Prima richiesta: miss, chiama il modello
        r1 = client.post("/v1/chat/completions", json=payload)
        assert r1.status_code == 200, r1.text
        assert r1.json()["choices"][0]["message"]["content"] == "risposta cached"

        # Seconda richiesta identica: hit da cache, il client NON viene chiamato di nuovo
        calls_before = len(fc.calls)
        r2 = client.post("/v1/chat/completions", json=payload)
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["choices"][0]["message"]["content"] == "risposta cached"
        assert body2["choices"][0]["finish_reason"] == "stop"
        assert len(fc.calls) == calls_before  # nessuna chiamata aggiuntiva

    def test_cache_contiene_dict_non_bytes(self):
        """TTLCache.get restituisce il dict salvato, non bytes."""
        cache = TTLCache(ttl_seconds=60)
        payload = {"choices": [{"message": {"content": "test"}}]}
        cache.set("k", payload)
        retrieved = cache.get("k")
        assert isinstance(retrieved, dict), f"atteso dict, ottenuto {type(retrieved)}"
        assert retrieved["choices"][0]["message"]["content"] == "test"

    def test_cache_non_causa_type_error(self):
        """JSONResponse(content=dict) non deve sollevare TypeError."""
        from fastapi.responses import JSONResponse
        d = {"id": "chatcmpl-test", "choices": []}
        resp = JSONResponse(content=d)
        assert resp.status_code == 200
        assert json.loads(resp.body)["id"] == "chatcmpl-test"


# ─── Bug (b): execute loop indent ─────────────────────────────────────────────

class TestExecuteLoopBug:
    """execute deve: (1) con args vuoti restituire il risultato del tool, non None;
    (2) validare TUTTI i tipi prima di eseguire; (3) eseguire il tool una sola volta."""

    def test_list_dir_senza_path_restituisce_stringa(self, ws):
        """list_dir ha path opzionale: execute({}) non deve restituire None."""
        (ws / "a.txt").write_text("x")
        t = _tb(ws)
        result = t.execute("list_dir", {})
        assert result is not None, "execute ha restituito None invece del risultato"
        assert isinstance(result, str), f"atteso str, ottenuto {type(result)}"
        assert "a.txt" in result

    def test_list_dir_args_vuoti_stringa(self, ws):
        """Anche con argomenti come stringa vuota, non deve restituire None."""
        t = _tb(ws)
        result = t.execute("list_dir", "{}")
        assert result is not None
        assert isinstance(result, str)

    def test_tipo_errato_su_secondo_argomento_non_esegue(self, ws):
        """Se il secondo argomento ha tipo sbagliato, l'errore viene restituito
        PRIMA di eseguire il tool (non a meta' esecuzione)."""
        (ws / "ok.txt").write_text("x")
        t = _tb(ws)
        # search_text ha: pattern (str, required), path (str), regex (bool)
        # Passiamo path come int: la validazione deve catturarlo
        result = t.execute("search_text", {"pattern": "x", "path": 123})
        assert "stringa" in result, f"atteso errore tipo, ottenuto: {result!r}"

    def test_execute_tool_chiamato_una_sola_volta(self, ws):
        """Con N argomenti validi il tool viene eseguito esattamente una volta."""
        calls = []
        original = Toolbox.list_dir

        class CountingToolbox(Toolbox):
            def list_dir(self, path="."):
                calls.append(path)
                return original(self, path)

        t = CountingToolbox(ws, {}, confirm=lambda _: True)
        result = t.execute("list_dir", {"path": "."})
        assert result is not None
        assert len(calls) == 1, f"tool chiamato {len(calls)} volte invece di 1"

    def test_execute_con_argomento_non_dict_restituisce_errore(self, ws):
        """Argomenti non-dict (e non stringa) restituiscono errore pulito."""
        t = _tb(ws)
        result = t.execute("list_dir", ["/tmp"])
        assert result is not None
        assert "errore" in result.lower() or "non validi" in result.lower()


# ─── Bug (c): .gitignore WAL/SHM ─────────────────────────────────────────────

class TestGitignoreWAL:
    """Verifica che i file WAL/SHM di SQLite siano nel .gitignore."""

    def test_gitignore_contiene_shm_e_wal(self):
        gitignore = Path(__file__).parent.parent / ".gitignore"
        assert gitignore.exists(), ".gitignore non trovato"
        content = gitignore.read_text(encoding="utf-8")
        assert "memoria.db-shm" in content, "memoria.db-shm mancante dal .gitignore"
        assert "memoria.db-wal" in content, "memoria.db-wal mancante dal .gitignore"
