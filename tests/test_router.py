import pytest

from conftest import conn_error, msg, rate_limit, tool_call
from tools import Toolbox

M1, M2 = "nvidia/nemotron-3.5-super-vl:free", "thinkingmachines/inkling-small:free"
GLOBAL = "openrouter/free"


def test_classificazione(make_router):
    r = make_router({})
    assert r.classify("C'è un bug nella funzione python") == "coding"
    assert r.classify("quanto rende un ETF?") == "finanza"
    assert r.classify("debugging del codice") == "coding"  # keyword con wildcard
    assert r.classify("ciao come stai") == "generale"
    assert r.classify("") == "generale"


def test_fallback_su_errore_e_risposta_vuota(make_router):
    r = make_router({M1: [conn_error()], M2: [msg("   ")], GLOBAL: [msg("ok")]})
    models = r.cfg["categories"]["generale"]["models"]
    assert models == [M1, M2]
    assert r._call(models, [{"role": "user", "content": "x"}]) == (GLOBAL, "ok")


def test_nessuna_scelta_e_tutti_falliscono(make_router):
    r = make_router({M1: ["no_choices"], M2: [conn_error()], GLOBAL: [msg(None)]})
    with pytest.raises(RuntimeError, match="Tutti i modelli"):
        r._call([M1, M2], [{"role": "user", "content": "x"}])


def test_429_rispetta_retry_after_poi_riprova(make_router):
    r = make_router({M1: [rate_limit(3), msg("fatto")]})
    assert r._call([M1], [{"role": "user", "content": "x"}]) == (M1, "fatto")
    assert make_router.sleeps == [3.0]


def test_429_senza_header_usa_backoff_e_ha_un_tetto(make_router):
    r = make_router({M1: [rate_limit()] * 3, GLOBAL: [msg("dal fallback")]})
    assert r._call([M1], [{"role": "user", "content": "x"}]) == (GLOBAL, "dal fallback")
    assert len(make_router.sleeps) == 2  # max_tentativi = 2, poi si passa oltre


def test_429_con_attesa_troppo_lunga_salta_subito(make_router):
    r = make_router({M1: [rate_limit(600)], GLOBAL: [msg("ok")]})
    assert r._call([M1], [{"role": "user", "content": "x"}])[0] == GLOBAL
    assert make_router.sleeps == []


def test_agente_con_tool_call(make_router, tmp_path):
    (tmp_path / "a.txt").write_text("ciao", encoding="utf-8")
    tb = Toolbox(tmp_path)
    r = make_router({M1: [msg(None, [tool_call("read_file", '{"path": "a.txt"}')]), msg("Contiene: ciao")]})
    cat, model, answer = r.ask("s", "leggi a.txt", "generale", toolbox=tb)
    assert (model, answer) == (M1, "Contiene: ciao")
    second = r.client.calls[1]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "ciao"}
    assert second[-2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert r.client.calls[0]["tools"]


def test_agente_argomenti_malformati_non_rompono(make_router, tmp_path):
    tb = Toolbox(tmp_path)
    r = make_router({M1: [msg(None, [tool_call("read_file", "{non json")]), msg("ok")]})
    r.ask("s", "x", "generale", toolbox=tb)
    assert r.client.calls[1]["messages"][-1]["content"].startswith("Errore")


def test_agente_si_ferma_al_massimo_dei_passi(make_router, tmp_path):
    loop = [msg(None, [tool_call("list_dir", "{}")])] * 3
    r = make_router({M1: loop}, strumenti={"max_passi": 3})
    _, _, answer = r.ask("s", "x", "generale", toolbox=Toolbox(tmp_path))
    assert "numero massimo di passi" in answer


def test_memoria_salva_e_riusa_lo_storico(make_router):
    r = make_router({M1: [msg("r1"), msg("r2")]})
    r.ask("s", "uno", "generale")
    r.ask("s", "due", "generale")
    sent = r.client.calls[1]["messages"]
    assert [m["content"] for m in sent[1:]] == ["uno", "r1", "due"]


def test_compattazione_riassume_e_non_spezza_le_coppie(make_router):
    r = make_router({M1: [msg("RIASSUNTO")] * 10}, memoria={"max_messaggi": 4})
    r.cfg["categories"]["appunti"]["models"] = [M1]
    for t in ("a", "b", "c"):
        r.ask("s", t, "generale")
    rows = r.memory.active("s")
    assert r.memory.summary("s") == "RIASSUNTO"
    assert rows[0][1] == "user" and len(rows) % 2 == 0
    assert len(rows) <= 4
    r.ask("s", "d", "generale")
    assert "RIASSUNTO" in r.client.calls[-1]["messages"][0]["content"]


def test_compattazione_fallita_non_perde_messaggi(make_router):
    r = make_router({M1: [msg("r"), msg("r"), msg(""), ], GLOBAL: [msg("")]}, memoria={"max_messaggi": 2})
    r.cfg["categories"]["appunti"]["models"] = [M1]
    r.ask("s", "a", "generale")
    r.ask("s", "b", "generale")
    assert r.memory.summary("s") == ""
    assert len(r.memory.active("s")) == 4
