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
    assert [e["hook_event_name"] for e in events] == ["UserPromptSubmit", "PreToolUse", "PreToolUse", "Stop"]
    assert events[0]["prompt"] == "ciao come stai"
    assert events[0]["session_title"] == "router · coding"
    assert events[1]["tool_name"] == "mcp__router__nvidia/nemotron:free"
    assert events[2]["tool_input"] == {"path": "a.py"}
    assert events[3]["last_assistant_message"] == "tutto ok"
    assert all(e["session_id"] == "router" for e in events)


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
