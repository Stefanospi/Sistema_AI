"""Ponte verso Capri (capri-desktop).

Invia a http://127.0.0.1:47823/event eventi nello stesso formato degli hook di Claude Code,
così Capri reagisce al router come se fosse Claude Code: ticker con il modello in uso,
festeggiamento a fine risposta, reazione agli errori.

Fire-and-forget: non rallenta mai il router; se Capri è spenta non succede nulla.
Tutto resta su 127.0.0.1. Per spegnerlo: CAPRI_ENABLED=0 nel .env.
"""
from __future__ import annotations

import http.client
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PORT = int(os.getenv("CAPRI_PORT") or 47823)
SESSION = "router"
CWD = str(Path(__file__).parent)

# un solo worker = gli eventi arrivano a Capri nell'ordine in cui li emettiamo
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capri")


def _enabled() -> bool:
    return os.getenv("CAPRI_ENABLED", "1") != "0"


def _clip(text, n: int) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[:n] + "…"


def _post(body: bytes) -> None:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=0.5)
        conn.request("POST", "/event", body, {"content-type": "application/json"})
        conn.getresponse().read()
        conn.close()
    except Exception:
        pass  # Capri spenta o non raggiungibile: va bene così


def emit(event: str, **fields) -> None:
    if not _enabled():
        return
    evt = {"hook_event_name": event, "session_id": SESSION, "cwd": CWD, **fields}
    try:
        _pool.submit(_post, json.dumps(evt, ensure_ascii=False).encode("utf-8"))
    except Exception:
        pass


# ---- eventi di alto livello -------------------------------------------------

def prompt(text: str, category: str | None = None) -> None:
    """Inizio richiesta (Capri si mette 'al lavoro')."""
    title = f"router · {category}" if category else "router"
    emit("UserPromptSubmit", prompt=_clip(" ".join(str(text).split()), 280), session_title=title)


def attempt(model: str) -> None:
    """Un tentativo su un modello: il ticker di Capri mostra 'router › <modello>'."""
    emit("PreToolUse", tool_name=f"mcp__router__{model}", tool_input={})


def tool(name: str, arguments: str | dict | None = None) -> None:
    """Uno strumento eseguito dall'agente (CLI --agente)."""
    args = arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    emit("PreToolUse", tool_name=name, tool_input=args if isinstance(args, dict) else {})


def done(answer: str) -> None:
    """Risposta completata (Capri festeggia e mostra l'inizio della risposta)."""
    emit("Stop", last_assistant_message=_clip(answer, 6000))


def failed(error: str) -> None:
    """Tutti i modelli hanno fallito."""
    emit("StopFailure", error=_clip(error, 280))
