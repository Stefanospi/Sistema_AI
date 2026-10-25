"""Server compatibile OpenAI (FastAPI + uvicorn) per il router AI.

Due modalità selezionabili tramite il campo "model" nella richiesta:
  - PROXY  (predefinita, per gli editor): il server sceglie il modello reale,
    inoltra messages/tools/tool_choice/stream SENZA modifiche, applica il fallback.
    La memoria del router è DISATTIVATA: l'editor gestisce già la sua cronologia.
  - AGENTE: espone le categorie come modelli virtuali (router-coding, router-auto…);
    gli strumenti di tools.py NON vengono esportati via HTTP (solo via CLI).

Sicurezza: ascolta solo su 127.0.0.1; token Bearer opzionale (ROUTER_API_TOKEN in .env);
nessuna chiave nei log; nessuna risposta di errore che riveli la chiave reale.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import APIError
from pydantic import BaseModel, Field

import capri_bridge as capri
from cache import response_cache, model_list_cache

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
CONFIG_PATH = BASE_DIR / "config.yaml"

load_dotenv(ENV_PATH)

log = logging.getLogger("airouter.server")

# ---------- avvio lazy del router ----------

_router_instance = None


def get_router():
    """Crea il Router una sola volta (pattern singleton con import lazy)."""
    global _router_instance
    if _router_instance is None:
        from router import Router
        _router_instance = Router()
    return _router_instance


# ---------- autenticazione opzionale ----------

_bearer = HTTPBearer(auto_error=False)


def check_token(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    """Se ROUTER_API_TOKEN è impostato in .env lo verifica; altrimenti lascia passare."""
    expected = os.getenv("ROUTER_API_TOKEN", "").strip()
    if not expected:
        return  # token non configurato: accesso libero (solo 127.0.0.1)
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Token non valido o mancante.")


# ---------- helper formato errore OpenAI ----------

def openai_error(status: int, message: str, err_type: str = "server_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": str(status)}},
    )


# ---------- nomi dei modelli virtuali ----------

VIRTUAL_PREFIX = "router-"
AUTO_MODEL = "router-auto"


def _category_from_model(model: str, categories: list[str]) -> str | None:
    """'router-coding' → 'coding'; 'router-auto' → None (classificazione automatica)."""
    if model == AUTO_MODEL:
        return None  # classifica da solo
    if model.startswith(VIRTUAL_PREFIX):
        cat = model[len(VIRTUAL_PREFIX):]
        if cat in categories:
            return cat
    return None  # non è un nome virtuale: il chiamante gestisce come proxy


# ---------- modelli Pydantic per validazione richieste ----------

class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model: str = AUTO_MODEL
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list | None = None
    tool_choice: Any = None
    stream: bool = False


# ---------- helper: risoluzione modelli (estrae da _proxy_complete e _proxy_stream) ----------

def _resolve_models(router, model: str, messages: list[dict]) -> tuple[list[str], str]:
    """Risolve lista modelli e categoria dal nome modello virtuale o reale.

    Restituisce (models_list, cat_name).
    """
    cats = list(router.cfg["categories"])
    cat_name = _category_from_model(model, cats)
    if cat_name is None:
        # Se il model è un nome reale OpenRouter, usalo direttamente; altrimenti classifica
        if "/" in model:
            return [model], "generale"
        # router-auto o nome non riconosciuto: classifica
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if isinstance(last_user, list):
            last_user = " ".join(p.get("text", "") for p in last_user if isinstance(p, dict))
        cat_name = router.classify(str(last_user))
        return router.cfg["categories"][cat_name]["models"], cat_name
    return router.cfg["categories"][cat_name]["models"], cat_name


def _last_user_text(messages: list[dict]) -> str:
    """Ultimo messaggio utente come testo (serve a Capri per mostrare cosa stai chiedendo)."""
    last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if isinstance(last, list):
        last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
    return str(last or "")


# ---------- helper: chiave cache per richiesta ----------

def _cache_key(body: dict) -> str:
    """Genera chiave deterministica per cache (esclude stream)."""
    payload = {
        "model": body.get("model"),
        "messages": body.get("messages"),
        "tools": body.get("tools"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "resp:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ---------- costruzione risposta non-stream ----------

def _build_response(model_used: str, content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _build_chunk(model_used: str, content: str | None, finish_reason: str | None = None) -> str:
    """Chunk SSE per lo streaming."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content} if content is not None
                else {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ---------- modalità PROXY ----------

def _proxy_complete(router, messages: list[dict], model: str,
                    tools: list | None, tool_choice: Any) -> JSONResponse:
    """Proxy non-stream: sceglie il modello reale e inoltra la richiesta invariata."""
    # Chiave cache (solo se non ci sono tool, per evitare effetti collaterali)
    cache_key = None
    if not tools:
        cache_key = _cache_key({
            "model": model,
            "messages": messages,
            "tools": tools,
        })
        cached = response_cache.get(cache_key)
        if cached is not None:
            return JSONResponse(content=cached)

    models_list, cat_name = _resolve_models(router, model, messages)
    capri.prompt(_last_user_text(messages), cat_name)

    tool_schemas = tools if tools else None
    try:
        used_model, msg_obj = router._complete(
            models_list, messages, tool_schemas, category=cat_name
        )
    except RuntimeError as e:
        capri.failed(str(e))
        return openai_error(502, str(e))
    except APIError as e:
        capri.failed(str(e))
        return openai_error(getattr(e, "status_code", 502) or 502, str(e))
    capri.done(getattr(msg_obj, "content", "") or "")

    # Gestione tool_calls nella risposta
    if getattr(msg_obj, "tool_calls", None):
        tc_list = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments or "{}",
                },
            }
            for c in msg_obj.tool_calls
        ]
        response = JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": used_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": tc_list},
                "finish_reason": "tool_calls",
                "logprobs": None,
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
    else:
        content = (msg_obj.content or "").strip()
        response = JSONResponse(_build_response(used_model, content))

    # Salva in cache se applicabile (nessuna tool_call)
    if cache_key is not None and not getattr(msg_obj, "tool_calls", None):
        try:
            response_cache.set(cache_key, response.body if hasattr(response, 'body') else json.dumps(response.content).encode())
        except Exception:
            pass  # cache best-effort

    return response


async def _proxy_stream(router, messages: list[dict], model: str,
                        tools: list | None) -> AsyncGenerator[str, None]:
    """Proxy stream: genera SSE, con fallback prima del primo token."""
    models_list, cat_name = _resolve_models(router, model, messages)
    capri.prompt(_last_user_text(messages), cat_name)

    tool_schemas = tools if tools else None
    used_model = model  # aggiornato al primo "start"
    parts: list[str] = []
    failed = False
    try:
        for ev in router._stream(models_list, messages, tool_schemas, category=cat_name):
            if ev[0] == "start":
                used_model = ev[1]
            elif ev[0] == "text":
                parts.append(ev[1])
                yield _build_chunk(used_model, ev[1])
            elif ev[0] == "final":
                used_model = ev[1]
                final_msg = ev[2]
                # Se la risposta finale contiene tool_calls (stream), emettiamo chunk speciali
                if getattr(final_msg, "tool_calls", None):
                    for tc in final_msg.tool_calls:
                        tc_chunk = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": used_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments or "{}",
                                        },
                                    }]
                                },
                                "finish_reason": None,
                                "logprobs": None,
                            }],
                        }
                        yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"
                    yield _build_chunk(used_model, None, "tool_calls")
                else:
                    yield _build_chunk(used_model, None, "stop")
    except RuntimeError as e:
        failed = True
        capri.failed(str(e))
        err_chunk = {
            "error": {"message": str(e), "type": "server_error", "code": "502"}
        }
        yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
    finally:
        # anche se l'editor chiude la connessione a metà, Capri non resta "al lavoro"
        if not failed:
            capri.done("".join(parts))
    yield "data: [DONE]\n\n"


# ---------- app FastAPI ----------

app = FastAPI(
    title="AI Router",
    description="Router di modelli AI gratuiti su OpenRouter, compatibile con le API OpenAI.",
    version="1.0.0",
    docs_url=None,   # disabilita Swagger UI in produzione locale
    redoc_url=None,
)


@app.get("/v1/models", dependencies=[Depends(check_token)])
def list_models():
    """Elenca i modelli virtuali del router (uno per categoria + router-auto)."""
    router = get_router()
    cats = list(router.cfg["categories"])
    now = int(time.time())
    models = []
    # router-auto: scelta automatica della categoria
    models.append({
        "id": AUTO_MODEL,
        "object": "model",
        "created": now,
        "owned_by": "ai-router",
        "description": "Classifica automaticamente la richiesta e sceglie il modello migliore.",
    })
    # un modello virtuale per ogni categoria
    for cat in cats:
        models.append({
            "id": f"{VIRTUAL_PREFIX}{cat}",
            "object": "model",
            "created": now,
            "owned_by": "ai-router",
            "description": f"Usa i modelli della categoria '{cat}'.",
        })
    return {"object": "list", "data": models}


@app.get("/health", dependencies=[Depends(check_token)])
def health():
    """Health check del server."""
    router = get_router()
    return {
        "status": "ok",
        "timestamp": int(time.time()),
        "version": "1.0.0",
        "host": "127.0.0.1",
        "cache": response_cache.stats(),
    }


@app.get("/stats", dependencies=[Depends(check_token)])
def stats():
    """Statistiche di utilizzo (cache, router, configurazione)."""
    router = get_router()
    cats = list(router.cfg["categories"])
    return {
        "cache": response_cache.stats(),
        "model_list_cache": model_list_cache.stats(),
        "categories": cats,
        "fallback_globale": router.cfg.get("fallback_globale"),
        "virtual_models": ["router-auto"] + [f"router-{c}" for c in cats],
    }


@app.post("/cache/clear", dependencies=[Depends(check_token)])
def clear_cache():
    """Svuota la cache delle risposte."""
    cleared = response_cache.clear()
    return {"cleared": cleared, "message": f"Cache svuotata: {cleared} entrate"}


@app.post("/v1/chat/completions", dependencies=[Depends(check_token)])
async def chat_completions(request: Request):
    """
    Endpoint principale, compatibile OpenAI.

    Modalità:
    - PROXY (default): messages/tools/stream vengono inoltrati al modello reale senza alterazioni.
      La memoria del router è disabilitata; lo storico è gestito dall'editor.
    - Le categorie vengono selezionate tramite il campo 'model' della richiesta
      (es. 'router-coding', 'router-auto').
    """
    try:
        body = await request.json()
    except Exception:
        return openai_error(400, "Corpo della richiesta non valido (JSON malformato).", "invalid_request_error")

    # Validazione Pydantic
    try:
        validated = ChatCompletionRequest(**body)
    except Exception as e:
        return openai_error(400, f"Validazione richiesta fallita: {str(e)}", "invalid_request_error")

    model: str = validated.model
    messages: list[dict] = [m.model_dump() for m in validated.messages]
    tools: list | None = validated.tools
    stream: bool = validated.stream

    if not messages:
        return openai_error(400, "Il campo 'messages' è obbligatorio e non può essere vuoto.", "invalid_request_error")

    router = get_router()

    if stream:
        return StreamingResponse(
            _proxy_stream(router, messages, model, tools),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return _proxy_complete(router, messages, model, tools, body.get("tool_choice"))


# ---------- avvio diretto ----------

def main():
    """Punto di ingresso per avviare il server da riga di comando."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    # Sicurezza: non si può forzare 0.0.0.0 o altri host via config
    if host != "127.0.0.1":
        print(f"Attenzione: host '{host}' non consentito per sicurezza; uso 127.0.0.1.")
        host = "127.0.0.1"
    port = int(server_cfg.get("porta", 8000))
    reload = bool(server_cfg.get("reload", False))

    print(f"Avvio server AI Router su http://{host}:{port}")
    print(f"  Modelli virtuali: GET http://{host}:{port}/v1/models")
    print(f"  Completamenti:   POST http://{host}:{port}/v1/chat/completions")
    token = os.getenv("ROUTER_API_TOKEN", "").strip()
    if token:
        print("  Autenticazione Bearer: ATTIVA (ROUTER_API_TOKEN impostato)")
    else:
        print("  Autenticazione Bearer: disattivata (server locale, solo 127.0.0.1)")
    print("  Premi Ctrl+C per fermare.\n")

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",   # uvicorn mostra solo warning/error; i nostri log vanno in router.log
    )


if __name__ == "__main__":
    main()
