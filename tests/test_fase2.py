import base64

import pytest

from conftest import StreamBreak, conn_error, msg, rate_limit, status_error, tool_call
from router import image_part
from tools import Toolbox

M1, M2 = "nvidia/nemotron-3.5-super-vl:free", "thinkingmachines/inkling-small:free"
LIGHT = "nvidia/nemotron-3.5-lightning:free"
GLOBAL = "openrouter/free"
USER = [{"role": "user", "content": "x"}]


# ---------- streaming ----------
def test_stream_emette_pezzi_e_modello(make_router):
    r = make_router({M1: [msg("ciao mondo")]})
    out, started = [], []
    cat, model, answer = r.ask("s", "ciao", "generale", on_text=out.append, on_model=started.append)
    assert "".join(out) == "ciao mondo" and len(out) == 2
    assert started == [M1] and (model, answer) == (M1, "ciao mondo")
    assert r.client.calls[0]["stream"] is True
    assert [m[2] for m in r.memory.active("s")] == ["ciao", "ciao mondo"]


def test_stream_fallback_prima_del_testo(make_router):
    r = make_router({M1: [conn_error()], M2: ["no_choices"], GLOBAL: [msg("ok")]})
    out = []
    _, model, _ = r.ask("s", "x", "generale", on_text=out.append)
    assert model == GLOBAL and "".join(out) == "ok"


def test_stream_interrotto_dopo_il_testo_non_ripete(make_router):
    r = make_router({M1: [StreamBreak(["par"], conn_error())], M2: [msg("altro")]})
    out = []
    with pytest.raises(RuntimeError, match="interrotta"):
        r.ask("s", "x", "generale", on_text=out.append)
    assert out == ["par"] and r.client.calls[-1]["model"] == M1
    assert r.memory.active("s") == []  # niente salvato su risposte parziali


def test_stream_con_tool_call(make_router, tmp_path):
    (tmp_path / "a.txt").write_text("ciao", encoding="utf-8")
    r = make_router({M1: [msg(None, [tool_call("read_file", '{"path": "a.txt"}')]), msg("Letto")]})
    out = []
    _, model, answer = r.ask("s", "x", "generale", Toolbox(tmp_path), on_text=out.append)
    assert answer == "Letto" and "".join(out) == "Letto"
    assert r.client.calls[1]["messages"][-1]["content"] == "ciao"


# ---------- tool nativi non supportati ----------
def test_modello_senza_tool_viene_saltato_e_ricordato(make_router, tmp_path):
    r = make_router({M1: [status_error(404)], M2: [msg(None, [tool_call("list_dir", "{}")]), msg("fine")]})
    r.cfg["categories"]["generale"]["models"] = [M1, M2]
    _, model, answer = r.ask("s", "x", "generale", Toolbox(tmp_path))
    assert (model, answer) == (M2, "fine")
    assert M1 in r._no_tools
    assert [c["model"] for c in r.client.calls] == [M1, M2, M2]  # al secondo passo M1 non viene riprovato


def test_senza_nessun_modello_con_tool_errore_chiaro(make_router, tmp_path):
    r = make_router({M1: [status_error(400)], M2: [status_error(400)], GLOBAL: [status_error(404)]})
    with pytest.raises(RuntimeError, match="Tutti i modelli"):
        r.ask("s", "x", "generale", Toolbox(tmp_path))
    with pytest.raises(RuntimeError, match="supporti gli strumenti"):
        r.ask("s", "x", "generale", Toolbox(tmp_path))


def test_400_senza_tool_non_blocca_il_modello(make_router):
    r = make_router({M1: [status_error(400), msg("ok")], M2: [msg("ok2")]})
    r.ask("s", "x", "generale")
    assert r._no_tools == set()


# ---------- classificatore LLM ----------
def test_classificatore_spento_di_default(make_router):
    r = make_router({})
    assert r.cfg["classificatore_llm"]["abilitato"] is False
    assert r.classify("qualcosa di vago") == "generale"
    assert r.client.calls == []


def test_classificatore_usato_solo_se_le_regole_falliscono(make_router):
    cfg = {"abilitato": True, "modello": LIGHT}
    r = make_router({LIGHT: [msg("Categoria: finanza.")]}, classificatore_llm=cfg)
    assert r.classify("c'è un bug nel codice") == "coding"  # regole: nessuna chiamata
    assert r.client.calls == []
    assert r.classify("dove metto i miei soldi?") == "finanza"
    assert [c["model"] for c in r.client.calls] == [LIGHT]  # niente fallback globale


def test_classificatore_risposta_inutile_o_errore_da_generale(make_router):
    cfg = {"abilitato": True, "modello": LIGHT}
    r = make_router({LIGHT: [msg("boh"), conn_error()]}, classificatore_llm=cfg)
    assert r.classify("frase vaga") == "generale"
    assert r.classify("altra frase vaga") == "generale"


# ---------- registro ----------
def test_log_senza_contenuti_sensibili(make_router):
    r = make_router({M1: [conn_error()], M2: [msg("RISPOSTA-SEGRETA")]})
    r.ask("s", "DOMANDA-SEGRETA", "generale")
    text = make_router.log_path.read_text(encoding="utf-8")
    assert "esito=errore:APIConnectionError" in text and f"modello={M2}" in text
    assert "categoria=generale" in text and "ms=" in text
    assert "SEGRETA" not in text and "sk-" not in text


def test_log_non_contiene_la_chiave(make_router, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-CHIAVE-DI-PROVA")
    r = make_router({M1: [rate_limit(1), msg("ok")]})
    r.ask("s", "x", "generale")
    assert "CHIAVE-DI-PROVA" not in make_router.log_path.read_text(encoding="utf-8")


# ---------- immagini ----------
def test_image_part(tmp_path):
    p = tmp_path / "s.png"
    p.write_bytes(b"\x89PNGdati")
    part = image_part(p)
    assert part["type"] == "image_url"
    assert part["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNGdati").decode()
    for bad, match in (("x.txt", "non supportato"), ("manca.png", "non trovata")):
        with pytest.raises(ValueError, match=match):
            image_part(tmp_path / bad)
    big = tmp_path / "big.png"
    big.write_bytes(b"0" * (5 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="troppo grande"):
        image_part(big)


def test_immagini_solo_desktop_e_solo_modelli_con_visione(make_router, tmp_path):
    img = tmp_path / "s.png"
    img.write_bytes(b"png")
    r = make_router({M1: [msg("vedo una finestra")]})
    r.cfg["categories"]["desktop"]["models"] = ["cohere/north-mini-code:free", M1]  # il primo è solo testo
    _, model, answer = r.ask("s", "cosa vedi?", "desktop", images=[img])
    assert model == M1 and [c["model"] for c in r.client.calls] == [M1]  # nessun fallback globale
    content = r.client.calls[0]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "cosa vedi?"} and content[1]["type"] == "image_url"
    assert "immagine" in r.memory.active("s")[0][2] and "base64" not in r.memory.active("s")[0][2]
    with pytest.raises(RuntimeError, match="solo nella categoria desktop"):
        r.ask("s", "x", "coding", images=[img])


def test_immagini_senza_modelli_visione_errore_chiaro(make_router, tmp_path):
    img = tmp_path / "s.png"
    img.write_bytes(b"png")
    r = make_router({}, visione={"modelli": []})
    with pytest.raises(RuntimeError, match="Nessun modello con visione"):
        r.ask("s", "x", "desktop", images=[img])
    assert r.client.calls == []


def test_senza_immagini_il_messaggio_resta_testo(make_router):
    r = make_router({M1: [msg("ok")]})
    r.ask("s", "ciao", "desktop")
    assert r.client.calls[0]["messages"][-1] == {"role": "user", "content": "ciao"}
