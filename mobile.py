"""Capri sul telefono: una pagina di chat (installabile come app) e due endpoint, /m/ask e /m/reset.

Sicurezza, perché il server resta su 127.0.0.1 e il telefono lo raggiunge da un tunnel privato
(Tailscale Serve, vedi README):
  - spento di default (config.yaml: mobile.abilitato) e senza MOBILE_TOKEN in .env non risponde;
  - il token è obbligatorio, lungo almeno MIN_TOKEN caratteri e diverso da ROUTER_API_TOKEN;
  - dopo MAX_FAILS token sbagliati si blocca tutto per LOCK_SECONDS;
  - un browser può chiamare solo dalla stessa origine della pagina (Origin = Host);
  - da qui si può solo chiacchierare con il cervello: nessun comando, nessun file, nessuno schermo.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
from typing import Callable
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

MIN_TOKEN = 24
MAX_FAILS = 5
LOCK_SECONDS = 300
SESSION = "mobile"

# Intestazioni che aggiungono i proxy davanti al server (Tailscale Serve, Funnel, cloudflared...).
# Una richiesta che le porta arriva da fuori: da fuori si vede solo /m.
PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "tailscale-user-login", "cf-connecting-ip", "forwarded")


def from_proxy(request: Request) -> bool:
    return any(h in request.headers for h in PROXY_HEADERS)


class AskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class _Guard:
    """Conta i token sbagliati: oltre il limite chiude la porta per un po'."""

    def __init__(self):
        self.fails: list[float] = []
        self.lock = threading.Lock()

    def locked(self, now: float) -> bool:
        with self.lock:
            self.fails = [t for t in self.fails if now - t < LOCK_SECONDS]
            return len(self.fails) >= MAX_FAILS

    def fail(self, now: float) -> None:
        with self.lock:
            self.fails.append(now)


guard = _Guard()


def make_router(ask: Callable[[str], str], reset: Callable[[], int], enabled: Callable[[], bool]) -> APIRouter:
    """`ask(text)` risponde con il testo di Capri e può sollevare RuntimeError (messaggio mostrabile)."""
    router = APIRouter(prefix="/m")

    def token() -> str:
        t = os.getenv("MOBILE_TOKEN", "").strip()
        if len(t) < MIN_TOKEN or t == os.getenv("ROUTER_API_TOKEN", "").strip():
            return ""
        return t

    def active() -> bool:
        return enabled() and bool(token())

    def authorize(request: Request) -> None:
        if not active():
            raise HTTPException(status_code=404)
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host", ""):
            raise HTTPException(status_code=403, detail="Origine non consentita.")
        now = time.monotonic()
        if guard.locked(now):
            raise HTTPException(status_code=429, detail="Troppi tentativi: riprova tra qualche minuto.")
        given = request.headers.get("x-capri-token", "")
        if not secrets.compare_digest(given.encode("utf-8"), token().encode("utf-8")):
            guard.fail(now)
            raise HTTPException(status_code=401, detail="Token non valido.")

    @router.get("", response_class=HTMLResponse)
    def page():
        if not active():
            raise HTTPException(status_code=404)
        return HTMLResponse(PAGE, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @router.get("/manifest.json")
    def manifest():
        if not active():
            raise HTTPException(status_code=404)
        return JSONResponse({
            "name": "Capri", "short_name": "Capri", "start_url": "/m", "scope": "/m",
            "display": "standalone", "background_color": "#0b0710", "theme_color": "#0b0710",
            "icons": [{"src": "data:image/svg+xml," + ICON, "sizes": "any", "type": "image/svg+xml", "purpose": "any"}],
        })

    @router.post("/ask")
    def ask_endpoint(req: AskRequest, request: Request):
        authorize(request)
        try:
            return {"text": ask(req.text.strip())}
        except RuntimeError as e:   # budget, quota, CLI: messaggi già scritti per l'utente
            return JSONResponse(status_code=502, content={"error": str(e)})

    @router.post("/reset")
    def reset_endpoint(request: Request):
        authorize(request)
        return {"cleared": reset()}

    return router


ICON = ("%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230b0710'/%3E"
        "%3Cpath d='M12 30 22 8l8 16 8-18 14 26-8 4-6-12-8 16-8-14-6 14z' fill='%23ff2152'/%3E"
        "%3Crect x='20' y='36' width='24' height='10' fill='%2335ff7a'/%3E%3C/svg%3E")

PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0710">
<link rel="manifest" href="/m/manifest.json">
<title>Capri</title>
<style>
:root { --ink:#0b0710; --panel:#17101f; --red:#ff2152; --green:#35ff7a; --violet:#7b4bff; --paper:#f1ecf6; --dim:#9a8fa8; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--ink); color: var(--paper);
  font: 16px/1.45 ui-monospace, "Cascadia Mono", Menlo, monospace; }
body { display: flex; flex-direction: column; padding: env(safe-area-inset-top) 0 env(safe-area-inset-bottom); }
header { display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-bottom: 2px solid var(--violet); background: var(--panel); }
header b { color: var(--red); letter-spacing: .18em; }
header span { color: var(--green); font-size: 12px; }
header button { margin-left: auto; }
#log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 88%; padding: 10px 12px; white-space: pre-wrap; word-wrap: break-word;
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px)); }
.me { align-self: flex-end; background: var(--violet); }
.capri { align-self: flex-start; background: var(--panel); border-left: 3px solid var(--green); }
.err { align-self: flex-start; background: #3a0d1a; border-left: 3px solid var(--red); }
.busy { color: var(--dim); animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: .35; } }
form { display: flex; gap: 8px; padding: 12px 16px; border-top: 2px solid var(--violet); background: var(--panel); }
input, textarea { flex: 1; min-width: 0; background: var(--ink); color: var(--paper); border: 1px solid var(--violet);
  padding: 12px; font: inherit; border-radius: 0; }
button { background: var(--red); color: var(--paper); border: 0; padding: 0 16px; font: inherit; font-weight: 700; min-height: 44px; }
button.ghost { background: none; border: 1px solid var(--dim); color: var(--dim); min-height: 32px; font-size: 12px; }
button:disabled { opacity: .5; }
#login { margin: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; max-width: 420px; width: 100%; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<header><b>CAPRI</b><span>UPLINK</span><button class="ghost" id="reset" hidden>nuova chat</button></header>
<div id="login">
  <div>Collegamento remoto. Inserisci il token del tuo PC.</div>
  <input id="token" type="password" autocomplete="current-password" placeholder="token" autocapitalize="off" autocorrect="off">
  <button id="enter" type="button" style="min-height:48px">Collega</button>
</div>
<div id="log" hidden></div>
<form id="form" hidden>
  <input id="text" autocomplete="off" placeholder="Chiedi a Capri…" maxlength="4000">
  <button id="send">Invia</button>
</form>
<script>
const $ = (id) => document.getElementById(id);
const KEY = 'capri-token';
let token = '';
try { token = localStorage.getItem(KEY) || ''; } catch {}

function add(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  $('log').appendChild(d);
  $('log').scrollTop = $('log').scrollHeight;
  return d;
}

function show() {
  const on = Boolean(token);
  $('login').hidden = on;
  $('log').hidden = !on;
  $('form').hidden = !on;
  $('reset').hidden = !on;
  if (on && !$('log').children.length) add('capri', 'Collegamento stabilito. Ehi, choom.');
}

async function call(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'content-type': 'application/json', 'x-capri-token': token }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (r.status === 401) { token = ''; try { localStorage.removeItem(KEY); } catch {} show(); }
  if (!r.ok) throw new Error(data.error || data.detail || 'Errore ' + r.status);
  return data;
}

$('enter').onclick = () => {
  token = $('token').value.trim();
  if (!token) return;
  try { localStorage.setItem(KEY, token); } catch {}
  $('token').value = '';
  show();
};

$('form').onsubmit = async (e) => {
  e.preventDefault();
  const text = $('text').value.trim();
  if (!text) return;
  $('text').value = '';
  $('send').disabled = true;
  add('me', text);
  const wait = add('capri busy', 'Capri sta pensando…');
  try {
    const { text: answer } = await call('/m/ask', { text });
    wait.className = 'msg capri';
    wait.textContent = answer || '…';
  } catch (err) {
    wait.className = 'msg err';
    wait.textContent = err.message;
  }
  $('send').disabled = false;
  $('text').focus();
};

$('reset').onclick = async () => {
  try { await call('/m/reset'); } catch {}
  $('log').replaceChildren();
  show();
};

show();
</script>
</body>
</html>
"""
