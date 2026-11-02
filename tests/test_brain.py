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
from conftest import REAL_FIND_CLAUDE_BIN, msg  # noqa: E402

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
        REAL_FIND_CLAUDE_BIN("")


# ---------- Budget ----------

def test_budget_conta_e_blocca(tmp_path):
    b = Budget(tmp_path / "u.db", daily_usd=0.01)
    b.check()
    b.record("claude/haiku", 0.006)
    b.check()
    b.record("claude/haiku", 0.006)
    with pytest.raises(BudgetExceeded):
        b.check()
    assert b.status()["remaining_today_usd"] == 0


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
        self.events, self.error, self.aliases, self.efforts, self.sent = events, error, [], [], []

    def stream(self, messages, alias, effort=None):
        self.aliases.append(alias)
        self.efforts.append(effort)
        self.sent.append(messages)
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


def test_handles_senza_gratuiti_prende_tutto_tranne_i_tool(make_brain):
    brain, _ = make_brain(StubBackend(ok_events()))
    assert brain.handles("brain-auto", None)
    assert brain.handles("router-coding", None)          # niente free: va a Claude
    assert brain.handles("google/gemma-4-31b-it:free", None)
    assert not brain.handles("brain-auto", [{"type": "function"}])
    brain.cfg["abilitato"] = False
    assert not brain.handles("brain-auto", None)


# ---------- Brain: esecuzione e ripiego ----------

def test_usa_l_alias_del_livello_e_registra_il_costo(make_brain):
    backend = StubBackend(ok_events("ciao", cost=0.004))
    brain, _ = make_brain(backend)
    model, message = brain.complete(USER, "brain-medio", "generale")
    assert backend.aliases == ["sonnet"] and message.content == "ciao"
    assert brain.budget.spent_today() == pytest.approx(0.004)


def test_errore_della_cli_ripiega_sul_router_gratuito_solo_se_consentito(make_brain):
    backend = StubBackend(error=BackendError("CLI assente"))
    brain, router = make_brain(backend, script={M1: [msg("dal gratuito")]}, cfg={"modelli_gratuiti": True})
    model, message = brain.complete(USER, "brain-auto", "generale")
    assert model == M1 and message.content == "dal gratuito"


def test_budget_finito_ripiega_senza_chiamare_claude(make_brain):
    backend = StubBackend(ok_events())
    brain, _ = make_brain(backend, script={M1: [msg("gratis")]}, daily=0.01, cfg={"modelli_gratuiti": True})
    brain.budget.record("claude/haiku", 0.5)
    model, message = brain.complete(USER, "brain-auto", "generale")
    assert backend.aliases == [] and message.content == "gratis"


def test_errore_dopo_il_primo_testo_non_ripiega(make_brain):
    class Breaks:
        def stream(self, messages, alias, effort=None):
            yield ("start", "claude/haiku")
            yield ("text", "inizio")
            raise BackendError("interrotto")

    brain, router = make_brain(Breaks(), script={M1: [msg("non deve servire")]}, cfg={"modelli_gratuiti": True})
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
        assert "spent_today_usd" in client.get("/stats").json()["brain"]
    finally:
        server._router_instance = server._brain_instance = None
        server.response_cache.clear()


# ---------- politica: niente modelli gratuiti, privacy, consumi ----------

def test_handles_con_gratuiti_consentiti_solo_brain(make_brain):
    brain, _ = make_brain(StubBackend(ok_events()), cfg={"modelli_gratuiti": True})
    assert brain.handles("brain-basso", None)
    assert not brain.handles("router-coding", None)


def test_senza_gratuiti_l_errore_non_ripiega_mai(make_brain):
    backend = StubBackend(error=BackendError("CLI assente"))
    brain, router = make_brain(backend, script={M1: [msg("non deve partire")]})
    with pytest.raises(BackendError, match="CLI assente"):
        brain.complete(USER, "brain-auto", "generale")
    assert router.client.calls == []


def test_categoria_sensibile_non_va_mai_sui_gratuiti(make_brain):
    backend = StubBackend(error=BackendError("CLI assente"))
    brain, router = make_brain(backend, script={M1: [msg("no")]}, cfg={"modelli_gratuiti": True})
    assert brain.allow_free("generale") and not brain.allow_free("finanza")
    with pytest.raises(BackendError):
        brain.complete(USER, "brain-auto", "finanza")
    assert router.client.calls == []


def test_router_rifiuta_i_gratuiti_se_disattivati(make_router):
    r = make_router({M1: [msg("no")]})
    r.cfg["brain"] = {"modelli_gratuiti": False}
    with pytest.raises(RuntimeError, match="Modelli gratuiti disattivati"):
        r._complete([M1], USER)
    assert r.client.calls == []


def test_effort_segue_il_livello(make_brain):
    backend = StubBackend(ok_events())
    brain, _ = make_brain(backend)
    brain.complete(USER, "brain-basso", "generale")
    brain.complete(USER, "brain-alto", "generale")
    assert backend.efforts == ["low", "high"]


def test_quota_del_piano_blocca_sopra_soglia(make_brain):
    import time
    backend = StubBackend(ok_events())
    brain, _ = make_brain(backend, cfg={"quota_piano": {"finestra_5h_max": 0.5}})
    brain.budget.record_plan({"five_hour": {"utilization": 0.62, "resets_at": time.time() + 3600}})
    with pytest.raises(brain_mod.QuotaExceeded, match="62%"):
        brain.complete(USER, "brain-auto", "generale")
    assert backend.aliases == []


def test_quota_scaduta_non_blocca(make_brain):
    import time
    backend = StubBackend(ok_events())
    brain, _ = make_brain(backend)
    brain.budget.record_plan({"five_hour": {"utilization": 0.99, "resets_at": time.time() - 10}})
    assert brain.complete(USER, "brain-auto", "generale")[1].content


def test_dopo_la_risposta_salva_la_quota_letta(make_brain):
    import time
    from types import SimpleNamespace
    plan = {"five_hour": {"utilization": 0.3, "resets_at": time.time() + 600}}
    events = [("start", "claude/x"), ("text", "ok"),
              ("final", "claude/x", SimpleNamespace(content="ok", tool_calls=None, plan=plan,
                                                     usage={"cost_usd": 0.001, "input_tokens": 1, "output_tokens": 1}))]
    brain, _ = make_brain(StubBackend(events))
    brain.complete(USER, "brain-auto", "generale")
    assert brain.budget.plan_state()["five_hour"]["utilization"] == 0.3


def test_budget_settimanale(tmp_path):
    b = Budget(tmp_path / "u.db", daily_usd=0, weekly_usd=0.01)
    b.record("m", 0.02)
    with pytest.raises(BudgetExceeded, match="settimanale"):
        b.check()


def test_prepare_taglia_storico_e_aggiunge_stile(make_brain):
    brain, _ = make_brain(StubBackend(ok_events()), cfg={"limiti": {"max_messaggi_storico": 3}})
    msgs = [{"role": "system", "content": "Sei un assistente"}]
    for i in range(5):
        msgs += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs.append({"role": "user", "content": "ultima"})
    out = brain.prepare(msgs)
    assert out[0]["role"] == "system" and "Sei un assistente" in out[0]["content"] and "conciso" in out[0]["content"]
    assert out[-1]["content"] == "ultima"
    assert out[1]["role"] != "assistant"           # niente risposta orfana in testa
    assert len(out) - 1 <= 3


def test_prepare_rispetta_il_limite_di_caratteri(make_brain):
    brain, _ = make_brain(StubBackend(ok_events()), cfg={"limiti": {"max_caratteri_prompt": 100}})
    msgs = [{"role": "user", "content": "x" * 80}, {"role": "assistant", "content": "y" * 80},
            {"role": "user", "content": "ultima domanda"}]
    out = brain.prepare(msgs)
    assert out[-1]["content"] == "ultima domanda" and len(out) <= 3
    big = brain.prepare([{"role": "user", "content": "z" * 1000}])
    assert len(big[-1]["content"]) < 400 and "omessi" in big[-1]["content"]


def test_prepare_viene_applicato_prima_del_backend(make_brain):
    backend = StubBackend(ok_events())
    brain, _ = make_brain(backend)
    brain.complete([{"role": "user", "content": "ciao"}], "brain-auto", "generale")
    assert backend.sent[0][0]["role"] == "system" and "conciso" in backend.sent[0][0]["content"]


# ---------- backend: quota, effort, tetto di output ----------

def rate(five=0.07, seven=0.03):
    return json.dumps({"type": "rate_limit_event", "rate_limit_info": {"unifiedWindows": {
        "five_hour": {"utilization": five, "resetsAt": 1793669400, "status": "allowed"},
        "seven_day": {"utilization": seven, "resetsAt": 1793772000, "status": "allowed"}}}}) + "\n"


def test_backend_legge_la_quota_dal_flusso():
    events = list(backend_with(FakeProc([rate(0.07, 0.03), delta("ok"), result("ok")])).stream(USER, "haiku"))
    plan = events[-1][2].plan
    assert plan["five_hour"] == {"utilization": 0.07, "resets_at": 1793669400.0}
    assert plan["seven_day"]["utilization"] == 0.03


def test_backend_effort_e_tetto_output():
    seen = []
    list(backend_with(FakeProc([delta("x"), result("x")]), seen).stream(USER, "haiku", effort="low"))
    argv, env = seen[0]["argv"], seen[0]["env"]
    assert argv[argv.index("--effort") + 1] == "low"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "1500"


def test_backend_api_error_del_tetto_diventa_errore():
    proc = FakeProc([result("API Error: Claude's response exceeded the 1500 output token maximum.")])
    with pytest.raises(BackendError, match="exceeded"):
        list(backend_with(proc).stream(USER, "haiku"))


# ---------- server: router-* senza gratuiti va a Claude ----------

def test_server_router_coding_senza_gratuiti_usa_claude(make_brain):
    import server
    from fastapi.testclient import TestClient

    backend = StubBackend(ok_events("da claude"))
    brain, router = make_brain(backend, script={M1: [msg("NON deve servire")]})
    server._router_instance, server._brain_instance = router, brain
    server.response_cache.clear()
    try:
        client = TestClient(server.app, raise_server_exceptions=False)
        r = client.post("/v1/chat/completions", json={"model": "router-coding", "messages": USER})
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "da claude"
        assert router.client.calls == []
        # con strumenti: la CLI non puo' e i gratuiti sono spenti -> errore chiaro, nessuna fuga
        router.cfg["brain"] = {"modelli_gratuiti": False}
        t = client.post("/v1/chat/completions", json={
            "model": "router-coding", "messages": USER,
            "tools": [{"type": "function", "function": {"name": "x", "description": "x", "parameters": {}}}]})
        assert t.status_code == 502 and "gratuiti disattivati" in t.json()["error"]["message"]
        assert router.client.calls == []
    finally:
        server._router_instance = server._brain_instance = None
        server.response_cache.clear()


# ---------- fase 2: sessioni lato server, riassunti, CLI ----------

def contents(messages):
    return [(m["role"], m["content"]) for m in messages]


@pytest.fixture
def served(make_brain):
    import server
    from fastapi.testclient import TestClient

    def start(backend, **kw):
        brain, router = make_brain(backend, **kw)
        server._router_instance, server._brain_instance = router, brain
        server.response_cache.clear()
        return TestClient(server.app, raise_server_exceptions=False), brain, router

    yield start
    server._router_instance = server._brain_instance = None
    server.response_cache.clear()


def test_sessione_ricorda_gli_scambi(served):
    backend = StubBackend(ok_events("prima risposta"))
    client, brain, router = served(backend)
    post = lambda text, **kw: client.post("/v1/chat/completions", json={
        "model": "brain-basso", "session": "capri", "messages": [{"role": "user", "content": text}], **kw})
    assert post("prima domanda").status_code == 200
    backend.events = ok_events("seconda risposta")
    assert post("seconda domanda").status_code == 200
    sent = [m for m in backend.sent[1] if m["role"] != "system"]
    assert contents(sent) == [("user", "prima domanda"), ("assistant", "prima risposta"), ("user", "seconda domanda")]
    assert [r for _, r, _ in router.memory.active("capri")] == ["user", "assistant", "user", "assistant"]


def test_sessione_in_streaming_e_reset(served):
    backend = StubBackend(ok_events("risposta stream"))
    client, brain, router = served(backend)
    body = {"model": "brain-basso", "session": "capri", "stream": True,
            "messages": [{"role": "user", "content": "ciao"}]}
    s = client.post("/v1/chat/completions", json=body)
    assert "risposta stream" in s.text
    assert [(r, c) for _, r, c in router.memory.active("capri")] == [("user", "ciao"), ("assistant", "risposta stream")]
    r = client.post("/v1/sessions/capri/reset")
    assert r.status_code == 200 and r.json()["cleared"] == 2
    assert router.memory.active("capri") == []


def test_sessione_non_usa_la_cache(served):
    backend = StubBackend(ok_events("x"))
    client, _, _ = served(backend)
    body = {"model": "brain-basso", "session": "s", "messages": [{"role": "user", "content": "ciao"}]}
    client.post("/v1/chat/completions", json=body)
    client.post("/v1/chat/completions", json=body)
    assert len(backend.aliases) == 2


def test_sessioni_separate(served):
    backend = StubBackend(ok_events("ok"))
    client, brain, router = served(backend)
    for sid in ("a", "b"):
        client.post("/v1/chat/completions", json={
            "model": "brain-basso", "session": sid, "messages": [{"role": "user", "content": f"da {sid}"}]})
    assert [c for _, _, c in router.memory.active("a")][0] == "da a"
    assert [c for _, _, c in router.memory.active("b")][0] == "da b"


def test_sessione_solo_con_i_modelli_claude(served):
    client, _, router = served(StubBackend(ok_events()), cfg={"modelli_gratuiti": True})
    r = client.post("/v1/chat/completions", json={
        "model": "router-coding", "session": "s", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400 and "session" in r.json()["error"]["message"]
    bad = client.post("/v1/chat/completions", json={
        "model": "brain-basso", "session": "../etc", "messages": [{"role": "user", "content": "x"}]})
    assert bad.status_code == 400


def test_riassunto_con_claude_basso(make_brain):
    backend = StubBackend(ok_events("RIASSUNTO"))
    brain, router = make_brain(backend)
    router.cfg["memoria"]["max_messaggi"] = 4
    for i in range(4):
        router.memory.add("s", "user", f"domanda {i}")
        router.memory.add("s", "assistant", f"risposta {i}")
    router._maybe_compact("s")
    assert router.memory.summary("s") == "RIASSUNTO"
    assert backend.aliases == ["haiku"]          # livello basso, mai i modelli gratuiti
    assert len(router.memory.active("s")) < 8


def test_memory_clear(make_router):
    r = make_router({})
    r.memory.add("s", "user", "a")
    r.memory.compact("s", 1, "sommario")
    assert r.memory.clear("s") == 1
    assert r.memory.active("s") == [] and r.memory.summary("s") == ""


def test_ask_della_cli_passa_dal_cervello(make_brain):
    backend = StubBackend(ok_events("dal cervello"))
    brain, router = make_brain(backend, script={M1: [msg("NON deve servire")]})
    seen = []
    category, model, answer = router.ask("default", "ciao", category="generale", on_text=seen.append)
    assert answer == "dal cervello" and "".join(seen) == "dal cervello"
    assert router.client.calls == [] and backend.aliases == ["haiku"]
    assert [r for _, r, _ in router.memory.active("default")] == ["user", "assistant"]


def test_ask_con_immagini_non_usa_i_gratuiti(make_brain, tmp_path):
    backend = StubBackend(ok_events())
    brain, router = make_brain(backend, script={M1: [msg("NON deve servire")]})
    router.cfg["brain"] = {"modelli_gratuiti": False}
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG")
    with pytest.raises(RuntimeError, match="Modelli gratuiti disattivati"):
        router.ask("default", "che finestra e'?", category="desktop", images=[img])
    assert router.client.calls == []
