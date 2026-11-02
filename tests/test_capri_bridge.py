"""Il ponte verso Capri: eventi nell'ordine giusto, formato hook di Claude Code, nessun errore se Capri è spenta."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import capri_bridge as capri


def _fake_capri():
    events = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["content-length"])
            events.append(json.loads(self.rfile.read(n)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, events


def _flush():
    capri._pool.submit(lambda: None).result(timeout=3)


def test_sequenza_eventi(monkeypatch):
    srv, events = _fake_capri()
    monkeypatch.setattr(capri, "PORT", srv.server_address[1])
    capri.prompt("ciao  come\nstai", "coding")
    capri.attempt("nvidia/nemotron:free")
    capri.tool("read_file", '{"path": "a.py"}')
    capri.done("tutto ok")
    _flush()
    srv.shutdown()
    assert [e["type"] for e in events] == ["request.start", "model.attempt", "tool.call", "done"]
    assert events[0]["prompt"] == "ciao come stai"
    assert events[0]["title"] == "router · coding"
    assert events[1]["model"] == "nvidia/nemotron:free"
    assert events[2]["name"] == "read_file" and events[2]["input"] == {"path": "a.py"}
    assert events[3]["text"] == "tutto ok"
    assert all(e["session_id"] == "router" and e["v"] == 1 for e in events)


def test_fallito_e_conferma(monkeypatch):
    srv, events = _fake_capri()
    monkeypatch.setattr(capri, "PORT", srv.server_address[1])
    capri.failed("boom")
    capri.confirm("Modificare a.py?")
    _flush()
    srv.shutdown()
    assert [(e["type"], e.get("error") or e.get("message")) for e in events] == [
        ("failed", "boom"), ("needs.confirm", "Modificare a.py?")]


def test_capri_spenta_non_rompe_nulla(monkeypatch):
    monkeypatch.setattr(capri, "PORT", 1)  # nessuno in ascolto
    capri.prompt("x")
    capri.failed("boom")
    _flush()  # non deve sollevare


def test_disattivabile(monkeypatch):
    srv, events = _fake_capri()
    monkeypatch.setattr(capri, "PORT", srv.server_address[1])
    monkeypatch.setenv("CAPRI_ENABLED", "0")
    capri.prompt("x")
    _flush()
    srv.shutdown()
    assert events == []
