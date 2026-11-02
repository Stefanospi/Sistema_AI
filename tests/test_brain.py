"""Backend Claude e cervello: sottoprocesso simulato, nessuna chiamata reale alla CLI."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import backends  # noqa: E402
import brain as brain_mod  # noqa: E402
from backends import BackendError, ClaudeCliBackend, build_prompt  # noqa: E402
from brain import Brain, Budget, BudgetExceeded  # noqa: E402
from conftest import msg  # noqa: E402

M1 = "nvidia/nemotron-3.5-super-vl:free"


# ---------- sottoprocesso finto ----------

class FakeStdin:
    def __init__(self):
        self.data, self.closed = "", False

    def write(self, s):
        self.data += s

    def close(self):
        self.closed = True


class FakeProc:
    def __init__(self, lines, returncode=0):
        self.stdin = FakeStdin()
        self.stdout = iter(lines)
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True


def delta(text):
    return json.dumps({"type": "stream_event", "event": {
        "type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}}) + "\n"


def result(text="ciao mondo", cost=0.001, is_error=False):
    return json.dumps({"type": "result", "subtype": "success", "is_error": is_error, "result": text,
                       "total_cost_usd": cost,
                       "usage": {"input_tokens": 2, "cache_creation_input_tokens": 10,
                                 "cache_read_input_tokens": 5, "output_tokens": 7}}) + "\n"


def backend_with(proc, record=None):
    def popen(argv, **kwargs):
        if record is not None:
            record.append({"argv": argv, **kwargs, "proc": proc})
        return proc

    return ClaudeCliBackend({"binario": sys.executable}, popen=popen)


USER = [{"role": "user", "content": "ciao"}]


# ---------- build_prompt ----------

def test_prompt_singolo_messaggio():
    system, prompt = build_prompt([{"role": "system", "content": "Sii breve"}] + USER)
    assert system == "Sii breve" and prompt == "ciao"


def test_prompt_con_storico_tiene_per_ultimo_il_messaggio_corrente():
    msgs = [{"role": "user", "content": "uno"}, {"role": "assistant", "content": "due"},
            {"role": "user", "content": "tre"}]
    _, prompt = build_prompt(msgs)
    assert prompt.index("uno") < prompt.index("due") < prompt.index("tre")
    assert prompt.rstrip().endswith("tre")


def test_prompt_immagini_omesse():
    _, prompt = build_prompt([{"role": "user", "content": [
        {"type": "text", "text": "guarda"}, {"type": "image_url", "image_url": {"url": "data:..."}}]}])
    assert "guarda" in prompt and "data:" not in prompt


# ---------- ClaudeCliBackend ----------

def test_stream_emette_eventi_e_usage():
    proc = FakeProc(["non json\n", delta("ci"), delta("ao"), result("ciao")])
    events = list(backend_with(proc).stream(USER, "haiku"))
    assert [e[0] for e in events] == ["start", "text", "text", "final"]
    assert events[0][1] == "claude/haiku"
    final = events[-1][2]
    assert final.content == "ciao"
    assert final.usage == {"cost_usd": 0.001, "input_tokens": 17, "output_tokens": 7}


def test_argv_solo_testo_e_senza_chiave_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-segreta")
    seen = []
    list(backend_with(FakeProc([delta("x"), result("x")]), seen).stream(
        [{"role": "system", "content": "Sii breve"}] + USER, "sonnet"))
    argv, env = seen[0]["argv"], seen[0]["env"]
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--tools") + 1] == ""
    assert "--safe-mode" in argv and "--no-session-persistence" in argv
    assert argv[argv.index("--system-prompt") + 1] == "Sii breve"
    assert "ANTHROPIC_API_KEY" not in env           # altrimenti fatturerebbe l'API
    assert seen[0]["proc"].stdin.data == "ciao" and seen[0]["proc"].stdin.closed


def test_senza_delta_usa_il_testo_del_risultato():
    events = list(backend_with(FakeProc([result("solo risultato")])).stream(USER, "haiku"))
    assert [e[0] for e in events] == ["start", "text", "final"]
    assert events[1][1] == "solo risultato"


def test_errore_della_cli_diventa_backend_error():
    proc = FakeProc([result("limite raggiunto", is_error=True)], returncode=1)
    with pytest.raises(BackendError, match="limite raggiunto"):
        list(backend_with(proc).stream(USER, "haiku"))


def test_nessun_risultato_riporta_l_output_della_cli():
    proc = FakeProc(["Error: not logged in\n"], returncode=1)
    with pytest.raises(BackendError, match="not logged in"):
        list(backend_with(proc).stream(USER, "haiku"))


def test_binario_assente(monkeypatch):
    monkeypatch.setattr(backends.shutil, "which", lambda _n: None)
    monkeypatch.setattr(backends.Path, "home", lambda: Path("Z:/non/esiste"))
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    with pytest.raises(BackendError, match="non trovata"):
        backends.find_claude_bin("")


# ---------- Budget ----------

def test_budget_conta_e_blocca(tmp_path):
    b = Budget(tmp_path / "u.db", daily_usd=0.01)
    b.check()
    b.record("claude/haiku", 0.006)
    b.check()
    b.record("claude/haiku", 0.006)
    with pytest.raises(BudgetExceeded):
        b.check()
    assert b.status()["remaining_usd"] == 0


def test_budget_riparte_il_giorno_dopo(tmp_path):
    now = [1_700_000_000.0]
    b = Budget(tmp_path / "u.db", daily_usd=0.01, clock=lambda: now[0])
    b.record("m", 0.02)
    with pytest.raises(BudgetExceeded):
        b.check()
    now[0] += 86_400 * 2
    b.check()


def test_budget_non_crea_file_se_non_usato(tmp_path):
    Budget(tmp_path / "u.db")
    assert not (tmp_path / "u.db").exists()


# ---------- Brain: policy ----------

class StubBackend:
    def __init__(self, events=None, error=None):
        self.events, self.error, self.aliases = events, error, []

    def stream(self, messages, alias):
        self.aliases.append(alias)
        if self.error:
            raise self.error
        yield from self.events


def ok_events(text="risposta", cost=0.002):
    from types import SimpleNamespace
    return [("start", "claude/x"), ("text", text),
            ("final", "claude/x", SimpleNamespace(content=text, tool_calls=None,
                                                   usage={"cost_usd": cost, "input_tokens": 1, "output_tokens": 1}))]


@pytest.fixture
def make_brain(make_router, tmp_path):
    def factory(backend, script=None, cfg=None, daily=1.0):
        router = make_router(script or {})
        cfg = cfg if cfg is not None else {}
        cfg.setdefault("budget", {"giornaliero_usd": daily})
        return Brain(router, backend=backend, budget=Budget(tmp_path / "u.db", daily), cfg=cfg), router

    return factory


def test_livello_di_default_e_basso(make_brain):
    brain, _ = make_brain(StubBackend(ok_events()))
    assert brain.level("brain-auto", "coding", USER) == "basso"


def test_livelli_forzati(make_brain):
    brain, _ = make_brain(StubBackend(ok_events()))
    assert brain.level("brain-medio", None, USER) == "medio"
    assert brain.level("brain-alto", None, USER) == "alto"


def test_per_categoria_e_salita(make_brain):
    brain, _ = make_brain(StubBackend(ok_events()),
                          cfg={"per_categoria": {"coding": "medio"}, "salita": ["pensaci bene"]})
    assert brain.level("brain-auto", "coding", USER) == "medio"
    assert brain.level("brain-auto", "coding",
                       [{"role": "user", "content": "Pensaci bene per favore"}]) == "alto"
    assert brain.level("brain-auto", "appunti",
                       [{"role": "user", "content": "pensaci bene"}]) == "medio"


def test_handles_solo_brain_senza_tool(make_brain):
    brain, _ = make_brain(StubBackend(ok_events()))
    assert brain.handles("brain-auto", None)
    assert not brain.handles("brain-auto", [{"type": "function"}])
    assert not brain.handles("router-auto", None)
    brain.cfg["abilitato"] = False
    assert not brain.handles("brain-auto", None)


# ---------- Brain: esecuzione e ripiego ----------

def test_usa_l_alias_del_livello_e_registra_il_costo(make_brain):
    backend = StubBackend(ok_events("ciao", cost=0.004))
    brain, _ = make_brain(backend)
    model, message = brain.complete(USER, "brain-medio", "generale")
    assert backend.aliases == ["sonnet"] and message.content == "ciao"
    assert brain.budget.spent_today() == pytest.approx(0.004)


def test_errore_della_cli_ripiega_sul_router_gratuito(make_brain):
    backend = StubBackend(error=BackendError("CLI assente"))
    brain, router = make_brain(backend, script={M1: [msg("dal gratuito")]})
    model, message = brain.complete(USER, "brain-auto", "generale")
    assert model == M1 and message.content == "dal gratuito"


def test_budget_finito_ripiega_senza_chiamare_claude(make_brain):
    backend = StubBackend(ok_events())
    brain, _ = make_brain(backend, script={M1: [msg("gratis")]}, daily=0.01)
    brain.budget.record("claude/haiku", 0.5)
    model, message = brain.complete(USER, "brain-auto", "generale")
    assert backend.aliases == [] and message.content == "gratis"


def test_errore_dopo_il_primo_testo_non_ripiega(make_brain):
    class Breaks:
        def stream(self, messages, alias):
            yield ("start", "claude/haiku")
            yield ("text", "inizio")
            raise BackendError("interrotto")

    brain, router = make_brain(Breaks(), script={M1: [msg("non deve servire")]})
    with pytest.raises(BackendError, match="interrotto"):
        list(brain.stream(USER, "brain-auto", "generale"))
    assert router.client.calls == []


# ---------- server: modelli brain-* ----------

def test_server_brain_non_stream_e_stream(make_brain):
    import server
    from fastapi.testclient import TestClient

    backend = StubBackend(ok_events("ciao da claude"))
    brain, router = make_brain(backend)
    server._router_instance, server._brain_instance = router, brain
    server.response_cache.clear()
    try:
        client = TestClient(server.app, raise_server_exceptions=False)
        body = {"model": "brain-basso", "messages": USER}
        r = client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "ciao da claude"
        assert backend.aliases == ["haiku"]

        backend.events = ok_events("a stream")
        s = client.post("/v1/chat/completions", json={**body, "stream": True})
        assert s.status_code == 200 and "a stream" in s.text and s.text.rstrip().endswith("[DONE]")

        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert {"brain-auto", "brain-basso", "brain-medio", "brain-alto"} <= set(ids)
        assert "spent_usd" in client.get("/stats").json()["brain"]
    finally:
        server._router_instance = server._brain_instance = None
        server.response_cache.clear()
