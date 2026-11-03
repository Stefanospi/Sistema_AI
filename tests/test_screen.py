"""Domande sullo schermo: prompt, lettura della risposta, immagini verso la CLI, endpoint."""
import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import screen  # noqa: E402
from backends import BackendError, last_user_images  # noqa: E402
from test_brain import FakeProc, StubBackend, backend_with, delta, make_brain, ok_events, result  # noqa: E402,F401

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()
ELEMENTS = [
    {"i": 1, "t": "Button", "n": "Salva", "a": "SaveButton", "e": True, "x": 100, "y": 50, "w": 80, "h": 30},
    {"i": 2, "t": "MenuItem", "n": "File | Nuovo\ncon a capo", "a": "", "e": True, "x": 10, "y": 10, "w": 40, "h": 20},
    {"i": 3, "t": "Button", "n": "", "a": "AvatarButton", "e": False, "x": 300, "y": 10, "w": 30, "h": 30},
]
WINDOW = {"title": "Senza titolo - Blocco note", "process": "Notepad", "rect": [0, 0, 800, 600]}


# ---------- prompt ----------

def test_build_messages_solo_controlli():
    msgs = screen.build_messages("Dove salvo?", WINDOW, ELEMENTS)
    assert msgs[0]["role"] == "system" and "SOLO con un oggetto JSON" in msgs[0]["content"]
    text = msgs[1]["content"][0]["text"]
    assert len(msgs[1]["content"]) == 1                      # nessuna immagine
    assert "Notepad — Senza titolo - Blocco note" in text
    assert "1|Button|Salva" in text
    assert "2|MenuItem|File Nuovo con a capo" in text       # | e a capo tolti: non rompono il formato
    assert "3|Button|[AvatarButton]|disattivato" in text    # senza nome si usa l'id di automazione
    assert text.rstrip().endswith("Domanda: Dove salvo?")
    assert "100" not in text                                 # le coordinate non servono al modello: meno token


def test_build_messages_con_screenshot():
    msgs = screen.build_messages("Cosa premo?", WINDOW, [], PNG, [1568, 900])
    assert msgs[1]["content"][0]["type"] == "image_url"      # l'immagine prima del testo
    text = msgs[1]["content"][1]["text"]
    assert "Nessun controllo leggibile" in text and "1568x900" in text


def test_elementi_malformati_scartati_e_limite():
    els = [{"i": "x"}, {"t": "Button"}, None] + [{"i": n, "t": "Button", "n": "b", "x": 0, "y": 0, "w": 1, "h": 1} for n in range(1, 500)]
    assert len(screen.normalize_elements([e for e in els if e is not None])) == screen.MAX_ELEMENTS


# ---------- risposta ----------

def test_parse_elemento_valido():
    out = screen.parse_answer('{"risposta": "Premi Salva.", "elemento": 1, "punto": null}', ELEMENTS)
    assert out["text"] == "Premi Salva."
    assert out["target"] == {"source": "uia", "id": 1, "name": "Salva", "rect": [100, 50, 80, 30]}


def test_parse_con_recinto_di_codice_e_testo_attorno():
    raw = 'Ecco:\n```json\n{"risposta": "Qui.", "elemento": 1}\n```'
    assert screen.parse_answer(raw, ELEMENTS)["target"]["id"] == 1


def test_parse_id_inventato_non_da_bersaglio():
    out = screen.parse_answer('{"risposta": "Boh", "elemento": 99}', ELEMENTS)
    assert out == {"text": "Boh", "target": None}
    assert screen.parse_answer('{"risposta": "x", "elemento": true}', ELEMENTS)["target"] is None


def test_parse_punto_solo_dentro_l_immagine():
    out = screen.parse_answer('{"risposta": "Là.", "elemento": null, "punto": [540, 300]}', [], [640, 360])
    assert out["target"] == {"source": "image", "x": 540.0, "y": 300.0}
    assert screen.parse_answer('{"risposta": "Là.", "punto": [900, 300]}', [], [640, 360])["target"] is None
    assert screen.parse_answer('{"risposta": "Là.", "punto": [5, 5]}', [], None)["target"] is None


def test_parse_non_json_resta_risposta_semplice():
    out = screen.parse_answer("Non lo so, scusa.", ELEMENTS)
    assert out == {"text": "Non lo so, scusa.", "target": None}
    assert screen.parse_answer("", ELEMENTS) == {"text": "", "target": None}


# ---------- immagini verso la CLI ----------

def shot_messages(url=PNG, text="dove?"):
    return [{"role": "system", "content": "Sii breve"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}]


def test_last_user_images_blocchi():
    blocks = last_user_images(shot_messages())
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(blocks[0]["source"]["data"]) == b"\x89PNG fake"


def test_solo_data_url_base64_e_limiti():
    with pytest.raises(BackendError, match="non valida"):
        last_user_images(shot_messages("https://example.com/a.png"))
    with pytest.raises(BackendError, match="non valida"):
        last_user_images(shot_messages("data:text/html;base64,AAAA"))
    many = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG}}] * 4}]
    with pytest.raises(BackendError, match="Troppe"):
        last_user_images(many)


def test_le_immagini_vecchie_non_si_rimandano():
    msgs = shot_messages() + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "e poi?"}]
    assert last_user_images(msgs) == []


def test_backend_manda_l_immagine_in_stream_json():
    seen = []
    list(backend_with(FakeProc([delta("x"), result("x")]), seen).stream(shot_messages(), "haiku"))
    argv = seen[0]["argv"]
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert argv[argv.index("--tools") + 1] == ""                       # nessuno strumento anche con le immagini
    sent = json.loads(seen[0]["proc"].stdin.data)
    assert sent["type"] == "user" and sent["message"]["role"] == "user"
    kinds = [c["type"] for c in sent["message"]["content"]]
    assert kinds == ["image", "text"] and sent["message"]["content"][1]["text"] == "dove?"
    assert "[allegato omesso]" not in seen[0]["proc"].stdin.data


def test_senza_immagini_resta_il_testo_semplice():
    seen = []
    list(backend_with(FakeProc([delta("x"), result("x")]), seen).stream([{"role": "user", "content": "ciao"}], "haiku"))
    assert "--input-format" not in seen[0]["argv"] and seen[0]["proc"].stdin.data == "ciao"


# ---------- endpoint ----------

@pytest.fixture
def served(make_brain):
    import server
    from fastapi.testclient import TestClient

    def start(backend, **kw):
        brain, router = make_brain(backend, **kw)
        server._router_instance, server._brain_instance = router, brain
        return TestClient(server.app, raise_server_exceptions=False), brain, router

    yield start
    server._router_instance = server._brain_instance = None


def request_body(**extra):
    return {"question": "Dove salvo?", "window": WINDOW, "elements": ELEMENTS, **extra}


def test_endpoint_risponde_con_il_bersaglio(served):
    backend = StubBackend(ok_events('{"risposta": "Premi Salva.", "elemento": 1, "punto": null}'))
    client, brain, router = served(backend)
    r = client.post("/v1/screen/ask", json=request_body())
    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "Premi Salva." and data["target"]["rect"] == [100, 50, 80, 30]
    assert backend.aliases == ["haiku"]                                 # livello basso di default
    sent = backend.sent[0]
    assert "1|Button|Salva" in sent[-1]["content"][0]["text"]


def test_endpoint_e_sempre_claude_anche_con_i_gratuiti_attivi(served):
    from conftest import msg
    backend = StubBackend(error=BackendError("quota"))
    client, brain, router = served(backend, script={"nvidia/nemotron-3.5-lightning:free": [msg("NO")]},
                                   cfg={"modelli_gratuiti": True, "instradamento": {"gratuito": ["desktop"]}})
    r = client.post("/v1/screen/ask", json=request_body())
    assert r.status_code == 502 and "quota" in r.json()["error"]["message"]
    assert router.client.calls == []                                    # lo schermo non va mai sui gratuiti


def test_endpoint_con_immagine_e_livello_configurato(served):
    backend = StubBackend(ok_events('{"risposta": "Là.", "elemento": null, "punto": [540, 300]}'))
    client, brain, router = served(backend, cfg={"schermo": {"livello": "medio"}})
    r = client.post("/v1/screen/ask", json=request_body(elements=[], image=PNG, image_size=[640, 360]))
    assert r.status_code == 200 and r.json()["target"] == {"source": "image", "x": 540.0, "y": 300.0}
    assert backend.aliases == ["sonnet"]
    assert backend.sent[0][-1]["content"][0]["type"] == "image_url"


def test_endpoint_validazione_e_browser(served):
    client, _, _ = served(StubBackend(ok_events()))
    assert client.post("/v1/screen/ask", json={"question": ""}).status_code == 422
    assert client.post("/v1/screen/ask", headers={"Origin": "https://evil.example"}, json=request_body()).status_code == 403
