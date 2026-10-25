"""Router di AI gratuite su OpenRouter, con memoria della conversazione (SQLite)."""
import base64
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from getpass import getpass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace

import httpx
import requests
import yaml
from cache import TTLCache
from dotenv import load_dotenv, set_key
from openai import APIError, OpenAI, RateLimitError

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"
DB_PATH = BASE_DIR / "memoria.db"
LOG_PATH = BASE_DIR / "router.log"
BASE_URL = "https://openrouter.ai/api/v1"

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".gif": "image/gif"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024

log = logging.getLogger("airouter")


def setup_logging(cfg: dict, path: Path | None = None):
    """Registro locale: solo metadati (categoria, modello, durata, tipo di errore), mai testi o chiavi."""
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    log.propagate = False
    log.setLevel(logging.INFO)
    lc = cfg.get("log") or {}
    if path is None and not lc.get("abilitato", True):
        log.addHandler(logging.NullHandler())
        return
    target = path or (BASE_DIR / lc.get("file", "router.log"))
    handler = RotatingFileHandler(target, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)


def image_part(path: str | Path) -> dict:
    """Immagine locale -> contenuto 'image_url' (data URL) per i messaggi con visione."""
    p = Path(path).expanduser()
    mime = IMAGE_TYPES.get(p.suffix.lower())
    if not mime:
        raise ValueError(f"Formato immagine non supportato (ammessi: {', '.join(IMAGE_TYPES)}).")
    if not p.is_file():
        raise ValueError(f"Immagine non trovata: {p}")
    if p.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Immagine troppo grande (massimo 5 MB).")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


# ---------- API key ----------
def get_api_key() -> str:
    """Legge la chiave da .env / variabile d'ambiente; se manca la chiede e la salva in .env."""
    load_dotenv(ENV_PATH)
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        key = getpass("Incolla la tua API key di OpenRouter (non verrà mostrata): ").strip()
        if not key:
            raise SystemExit("Chiave vuota: impossibile continuare.")
        ENV_PATH.touch(exist_ok=True)
        set_key(str(ENV_PATH), "OPENROUTER_API_KEY", key)
        print("Chiave salvata in .env (il file è già nel .gitignore).")
    return key


def all_model_ids() -> set[str]:
    """Elenco pubblico dei modelli OpenRouter (serve a controllare che quelli in config esistano ancora)."""
    r = requests.get(f"{BASE_URL}/models", timeout=30)
    r.raise_for_status()
    return {m["id"] for m in r.json()["data"]}


# ---------- Memoria ----------
class Memory:
    def __init__(self, path: Path = DB_PATH):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS messages(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 session TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                 category TEXT, ts REAL NOT NULL, summarized INTEGER DEFAULT 0)"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS summaries(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 session TEXT NOT NULL, content TEXT NOT NULL, ts REAL NOT NULL)"""
        )
        self.db.commit()

    def add(self, session: str, role: str, content: str, category: str | None = None):
        self.db.execute(
            "INSERT INTO messages(session, role, content, category, ts) VALUES(?,?,?,?,?)",
            (session, role, content, category, time.time()),
        )
        self.db.commit()

    def active(self, session: str) -> list[tuple[int, str, str]]:
        """Messaggi non ancora riassunti, dal più vecchio al più recente."""
        return self.db.execute(
            "SELECT id, role, content FROM messages WHERE session=? AND summarized=0 ORDER BY id",
            (session,),
        ).fetchall()

    def summary(self, session: str) -> str:
        row = self.db.execute(
            "SELECT content FROM summaries WHERE session=? ORDER BY id DESC LIMIT 1", (session,)
        ).fetchone()
        return row[0] if row else ""

    def compact(self, session: str, upto_id: int, new_summary: str):
        self.db.execute(
            "INSERT INTO summaries(session, content, ts) VALUES(?,?,?)",
            (session, new_summary, time.time()),
        )
        self.db.execute(
            "UPDATE messages SET summarized=1 WHERE session=? AND id<=?", (session, upto_id)
        )
        self.db.commit()


# ---------- Router ----------
class Router:
    def __init__(self, config_path: Path = BASE_DIR / "config.yaml", client=None, memory=None,
                 sleep=time.sleep, log_file: Path | None = None):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        setup_logging(self.cfg, log_file)
        self._no_tools: set[str] = set()  # modelli che hanno rifiutato i tool nativi (validi per questa esecuzione)
        # cache risposte: TTL 5 min, max 200 entrate
        self._response_cache = TTLCache(ttl_seconds=300, max_size=200)
        # client/memory/sleep iniettabili: servono ai test (nessuna rete, nessuna attesa reale)
        self.client = client or OpenAI(
            base_url=BASE_URL,
            api_key=get_api_key(),
            default_headers={"X-Title": "AI Router personale"},
        )
        self.memory = memory or Memory()
        self._sleep = sleep
        # WAL mode per SQLite: migliore concorrenza e performance
        self.memory.db.execute("PRAGMA journal_mode=WAL")
        self.memory.db.execute("PRAGMA synchronous=NORMAL")
        self.memory.db.commit()

    # -- helper cache --
    def _cache_key(self, payload: dict) -> str:
        """Genera chiave deterministica per cache delle risposte."""
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return "router:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    # -- configurazione --
    def check_config(self) -> list[str]:
        """Restituisce i modelli presenti in config ma non più elencati da OpenRouter."""
        ids = all_model_ids()
        used = {m for c in self.cfg["categories"].values() for m in c["models"]}
        used.add(self.cfg["fallback_globale"])
        return sorted(m for m in used if m not in ids)

    # -- classificazione: prima le regole, poi (opzionale) un piccolo modello --
    def _classify_rules(self, text: str) -> str | None:
        tokens = re.findall(r"\.?[\w#]+", text.lower())
        best, best_score = None, 0
        for name, cat in self.cfg["categories"].items():
            score = 0
            for kw in cat.get("keywords", []):
                if kw.endswith("*"):
                    score += any(t.startswith(kw[:-1]) for t in tokens)
                else:
                    score += kw in tokens
            if score > best_score:
                best, best_score = name, score
        return best

    def _classify_llm(self, text: str) -> str | None:
        c = self.cfg.get("classificatore_llm") or {}
        if not c.get("abilitato") or not c.get("modello"):
            return None
        names = list(self.cfg["categories"])
        prompt = (
            f"Classifica la richiesta in UNA sola categoria tra: {', '.join(names)}.\n"
            "Rispondi solo con il nome della categoria, senza altro testo.\n\n"
            f"Richiesta: {text[:500]}"
        )
        try:
            _, msg = self._complete([c["modello"]], [{"role": "user", "content": prompt}],
                                    use_global=False, category="classificatore")
        except RuntimeError:
            return None
        for token in re.findall(r"[a-zà-ù_]+", (msg.content or "").lower()):
            if token in names:
                return token
        return None

    def classify(self, text: str) -> str:
        return self._classify_rules(text) or self._classify_llm(text) or "generale"

    # -- chiamata con fallback --
    def _create(self, model: str, messages: list[dict], tools: list[dict] | None, stream: bool = False):
        """Una chiamata; sul 429 attende (Retry-After se presente) e riprova, con un tetto di tentativi."""
        rl = self.cfg.get("rate_limit", {})
        max_retries = int(rl.get("max_tentativi", 2))
        max_wait = float(rl.get("attesa_max_s", 30))
        kwargs = {"tools": tools} if tools else {}
        if stream:
            kwargs["stream"] = True
        for attempt in range(max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=model, messages=messages, timeout=90, **kwargs
                )
            except RateLimitError as e:
                if attempt == max_retries:
                    raise
                wait = self._retry_after(e)
                if wait is None:
                    wait = min(2.0 * 2**attempt, max_wait)
                if wait > max_wait:  # attesa troppo lunga: meglio passare al modello successivo
                    raise
                self._sleep(wait)

    @staticmethod
    def _retry_after(e: APIError) -> float | None:
        headers = getattr(getattr(e, "response", None), "headers", None) or {}
        try:
            value = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            return None
        return max(value, 0.0)

    def _candidates(self, models: list[str], tools, vision: bool, use_global: bool) -> list[str]:
        cands = list(models)
        if use_global and not vision:
            cands.append(self.cfg["fallback_globale"])
        cands = list(dict.fromkeys(cands))
        if vision:  # solo modelli verificati con input immagine (config: visione.modelli)
            allowed = set((self.cfg.get("visione") or {}).get("modelli", []))
            cands = [m for m in cands if m in allowed]
            if not cands:
                raise RuntimeError("Nessun modello con visione configurato per questa categoria.")
        if tools:
            cands = [m for m in cands if m not in self._no_tools]
            if not cands:
                raise RuntimeError("Nessun modello disponibile che supporti gli strumenti (tool nativi).")
        return cands

    def _failed(self, model: str, e: Exception, tools, errors: list, t0: float, category):
        status = getattr(e, "status_code", None)
        label = e.__class__.__name__ + (f" {status}" if status else "")
        # 400/404 con tool: il modello non li supporta, lo saltiamo per le richieste successive
        if tools and status in (400, 404):
            self._no_tools.add(model)
            label += " (tool nativi non supportati?)"
        errors.append(f"{model}: {label}")
        log.info("tentativo categoria=%s modello=%s esito=errore:%s ms=%d",
                 category, model, label, (time.monotonic() - t0) * 1000)

    def _complete(self, models: list[str], messages: list[dict], tools: list[dict] | None = None,
                  vision: bool = False, use_global: bool = True, category: str | None = None):
        """Prova i modelli in ordine; restituisce (modello, messaggio) con testo o tool_calls."""
        errors = []
        for model in self._candidates(models, tools, vision, use_global):
            t0 = time.monotonic()
            try:
                r = self._create(model, messages, tools)
                if not r.choices:
                    errors.append(f"{model}: risposta senza contenuto")
                    continue
                msg = r.choices[0].message
                if (msg.content or "").strip() or getattr(msg, "tool_calls", None):
                    log.info("tentativo categoria=%s modello=%s esito=ok ms=%d",
                             category, model, (time.monotonic() - t0) * 1000)
                    return model, msg
                errors.append(f"{model}: risposta vuota")
            except (APIError, ValueError, httpx.HTTPError) as e:  # 429, modello ritirato, rete, JSON non valido
                self._failed(model, e, tools, errors, t0, category)
        raise RuntimeError("Tutti i modelli hanno fallito:\n  " + "\n  ".join(errors))

    def _stream(self, models: list[str], messages: list[dict], tools=None, vision: bool = False,
                use_global: bool = True, category: str | None = None):
        """Generatore di eventi: ("start", modello), ("text", pezzo), infine ("final", modello, messaggio).
        Il fallback è possibile solo finché non è stato emesso testo; dopo, un errore interrompe la risposta."""
        errors = []
        for model in self._candidates(models, tools, vision, use_global):
            t0 = time.monotonic()
            emitted = False
            parts: list[str] = []
            calls: dict[int, dict] = {}
            try:
                for chunk in self._create(model, messages, tools, stream=True):
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if not emitted:
                            emitted = True
                            yield ("start", model)
                        parts.append(delta.content)
                        yield ("text", delta.content)
                    for tc in getattr(delta, "tool_calls", None) or []:
                        slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                        slot["id"] = tc.id or slot["id"]
                        if tc.function:
                            slot["name"] += tc.function.name or ""
                            slot["arguments"] += tc.function.arguments or ""
            except (APIError, ValueError, httpx.HTTPError) as e:
                self._failed(model, e, tools, errors, t0, category)
                if emitted:
                    raise RuntimeError(f"Risposta interrotta da {model}: {errors[-1].split(': ', 1)[1]}")
                continue
            content = "".join(parts)
            tool_calls = [
                SimpleNamespace(id=c["id"] or f"call_{i}",
                                function=SimpleNamespace(name=c["name"], arguments=c["arguments"]))
                for i, c in sorted(calls.items())
            ]
            if not content.strip() and not tool_calls:
                errors.append(f"{model}: risposta vuota")
                continue
            log.info("tentativo categoria=%s modello=%s esito=ok stream=1 ms=%d",
                     category, model, (time.monotonic() - t0) * 1000)
            if not emitted:
                yield ("start", model)
            yield ("final", model, SimpleNamespace(content=content, tool_calls=tool_calls or None))
            return
        raise RuntimeError("Tutti i modelli hanno fallito:\n  " + "\n  ".join(errors))

    def _step(self, models, messages, tools=None, vision=False, on_text=None, on_model=None, category=None):
        if on_text is None:
            return self._complete(models, messages, tools, vision, category=category)
        for ev in self._stream(models, messages, tools, vision, category=category):
            if ev[0] == "text":
                on_text(ev[1])
            elif ev[0] == "start" and on_model:
                on_model(ev[1])
            elif ev[0] == "final":
                return ev[1], ev[2]

    def _call(self, models: list[str], messages: list[dict]) -> tuple[str, str]:
        model, msg = self._complete(models, messages)
        return model, (msg.content or "").strip()

    # -- ciclo agente con strumenti --
    def _agent_loop(self, models: list[str], messages: list[dict], toolbox, vision=False,
                    on_text=None, on_model=None, category=None) -> tuple[str, str]:
        max_steps = int(self.cfg.get("strumenti", {}).get("max_passi", 8))
        model = ""
        for _ in range(max_steps):
            model, msg = self._step(models, messages, toolbox.schemas, vision, on_text, on_model, category)
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                return model, (msg.content or "").strip()
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                               for c in calls],
            })
            for c in calls:
                result = toolbox.execute(c.function.name, c.function.arguments)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
        return model, "Mi sono fermato: raggiunto il numero massimo di passi dell'agente."

    # -- domanda con memoria --
    def ask(self, session: str, text: str, category: str | None = None, toolbox=None,
            on_text=None, on_model=None, images: list | None = None) -> tuple[str, str, str]:
        """on_text/on_model attivano lo streaming; images (percorsi) solo nella categoria desktop."""
        category = category or self.classify(text)

        # Chiave cache (solo senza immagini, senza toolbox, senza streaming)
        cache_key = None
        if not images and not toolbox and on_text is None:
            cache_key = self._cache_key({
                "category": category,
                "text": text,
                "session": session,
            })
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                return category, cached["model"], cached["answer"]

        cat = self.cfg["categories"][category]
        if images and category != "desktop":
            raise RuntimeError("Le immagini sono supportate solo nella categoria desktop.")

        system = cat["system"].strip()
        summary = self.memory.summary(session)
        if summary:
            system += "\n\nRiassunto della conversazione precedente:\n" + summary

        user_content = text
        if images:
            user_content = [{"type": "text", "text": text}] + [image_part(p) for p in images]
        messages = [{"role": "system", "content": system}]
        messages += [{"role": r, "content": c} for _, r, c in self.memory.active(session)]
        messages.append({"role": "user", "content": user_content})

        vision = bool(images)
        t0 = time.monotonic()
        try:
            if toolbox is not None:
                model, answer = self._agent_loop(cat["models"], messages, toolbox, vision,
                                                 on_text, on_model, category)
            else:
                model, msg = self._step(cat["models"], messages, None, vision, on_text, on_model, category)
                answer = (msg.content or "").strip()
        except RuntimeError:
            log.info("richiesta categoria=%s esito=fallita ms=%d", category, (time.monotonic() - t0) * 1000)
            raise
        log.info("richiesta categoria=%s modello=%s esito=ok stream=%d agente=%d immagini=%d ms=%d",
                 category, model, on_text is not None, toolbox is not None, len(images or []),
                 (time.monotonic() - t0) * 1000)
        # in memoria solo testo: le immagini non vengono conservate
        self.memory.add(session, "user", text + (f"\n[{len(images)} immagine/i allegata/e]" if images else ""), category)
        self.memory.add(session, "assistant", answer, category)
        self._maybe_compact(session)

        # Salva in cache se applicabile
        if cache_key is not None:
            try:
                self._response_cache.set(cache_key, {"model": model, "answer": answer})
            except Exception:
                pass  # cache best-effort

        return category, model, answer

    # -- riassunto automatico dei messaggi vecchi --
    def _maybe_compact(self, session: str):
        rows = self.memory.active(session)
        if len(rows) <= self.cfg["memoria"]["max_messaggi"]:
            return
        old = rows[: len(rows) // 2]
        while old and old[-1][1] != "assistant":  # non spezzare una coppia domanda/risposta
            old.pop()
        if not old:
            return
        transcript = "\n".join(f"{role}: {content[:1500]}" for _, role, content in old)
        previous = self.memory.summary(session)
        prompt = (
            "Riassumi in italiano, in massimo 200 parole, fatti, decisioni e preferenze emersi "
            "in questa conversazione.\n\n"
            + (f"Riassunto precedente:\n{previous}\n\n" if previous else "")
            + f"Nuovi messaggi:\n{transcript}"
        )
        try:
            _, new_summary = self._call(
                self.cfg["categories"].get("appunti", self.cfg["categories"]["generale"])["models"],
                [{"role": "user", "content": prompt}],
            )
        except RuntimeError:
            return  # riprova alla prossima risposta
        if not new_summary:
            return
        self.memory.compact(session, old[-1][0], new_summary)
