"""Cervello: sceglie livello e modello Claude per ogni richiesta, tiene il budget, e se Claude non
risponde (budget finito, CLI assente, errore) ripiega sul router dei modelli gratuiti.

Modelli virtuali esposti dal server: brain-auto (livello deciso dalla policy), brain-basso,
brain-medio, brain-alto (livello forzato).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import capri_bridge as capri
from backends import BackendError, ClaudeCliBackend

BASE_DIR = Path(__file__).parent
log = logging.getLogger("airouter.brain")

PREFIX = "brain-"
AUTO = "brain-auto"
LEVELS = ["basso", "medio", "alto"]  # dal più economico al più capace

DEFAULTS = {
    "abilitato": True,
    "claude": {"binario": "", "timeout_s": 120, "max_per_richiesta_usd": 0.05},
    "livelli": {"basso": "haiku", "medio": "sonnet", "alto": "opus"},
    "livello_default": "basso",
    "per_categoria": {},
    "salita": [],
    "budget": {"giornaliero_usd": 1.0},
}


class BudgetExceeded(BackendError):
    """Tetto giornaliero raggiunto: si ripiega sui modelli gratuiti."""


class Budget:
    """Spesa giornaliera in dollari a prezzo di listino (come la riporta la CLI).
    Con l'abbonamento non è un addebito: serve da contatore di consumo e da freno."""

    def __init__(self, path: Path | None = None, daily_usd: float = 1.0, clock=time.time):
        self.path = path or BASE_DIR / "brain_usage.db"
        self.daily_usd = float(daily_usd or 0)
        self._clock = clock
        self._db = None  # aperto al primo uso: niente file creati se il cervello non serve
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS usage(
                     id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT NOT NULL, ts REAL NOT NULL,
                     model TEXT NOT NULL, cost REAL NOT NULL, tokens_in INTEGER, tokens_out INTEGER)""")
            self._db.commit()
        return self._db

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(self._clock()))

    def spent_today(self) -> float:
        with self._lock:
            row = self._conn().execute("SELECT COALESCE(SUM(cost),0) FROM usage WHERE day=?",
                                       (self._today(),)).fetchone()
        return float(row[0])

    def check(self) -> None:
        if self.daily_usd and self.spent_today() >= self.daily_usd:
            raise BudgetExceeded(f"Budget giornaliero raggiunto ({self.daily_usd:.2f} $).")

    def record(self, model: str, cost: float, tokens_in: int = 0, tokens_out: int = 0) -> None:
        with self._lock:
            db = self._conn()
            db.execute("INSERT INTO usage(day, ts, model, cost, tokens_in, tokens_out) VALUES(?,?,?,?,?,?)",
                       (self._today(), self._clock(), model, cost, tokens_in, tokens_out))
            db.commit()

    def status(self) -> dict:
        spent = self.spent_today()
        return {"spent_usd": round(spent, 5), "daily_usd": self.daily_usd,
                "remaining_usd": round(max(self.daily_usd - spent, 0.0), 5) if self.daily_usd else None}


def _merge(base: dict, over: dict | None) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def _last_user_text(messages: list[dict]) -> str:
    last = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    if isinstance(last, list):
        last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
    return str(last or "")


class Brain:
    def __init__(self, router, backend=None, budget: Budget | None = None, cfg: dict | None = None):
        self.router = router
        self.cfg = _merge(DEFAULTS, cfg if cfg is not None else router.cfg.get("brain"))
        self.backend = backend or ClaudeCliBackend(self.cfg["claude"])
        self.budget = budget or Budget(daily_usd=self.cfg["budget"]["giornaliero_usd"])

    # -- quali richieste gestisce --
    def handles(self, model: str, tools) -> bool:
        """Solo i modelli brain-* e solo senza tool: la CLI qui gira senza strumenti."""
        return bool(self.cfg["abilitato"]) and model.startswith(PREFIX) and not tools

    def virtual_models(self) -> list[str]:
        return [AUTO] + [PREFIX + lv for lv in LEVELS] if self.cfg["abilitato"] else []

    # -- policy: livello e modello --
    def level(self, model: str, category: str | None, messages: list[dict]) -> str:
        forced = model[len(PREFIX):]
        if forced in LEVELS:
            return forced
        level = self.cfg["per_categoria"].get(category) or self.cfg["livello_default"]
        if level not in LEVELS:
            level = "basso"
        text = _last_user_text(messages).lower()
        if any(k.lower() in text for k in self.cfg["salita"]):
            level = LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]
        return level

    def status(self) -> dict:
        return {**self.budget.status(), "livelli": self.cfg["livelli"],
                "livello_default": self.cfg["livello_default"]}

    # -- esecuzione --
    def stream(self, messages: list[dict], model: str, category: str | None = None):
        """Stessi eventi di Router._stream. Se Claude fallisce prima del primo testo, ripiega
        sul router gratuito della categoria; dopo il primo testo l'errore interrompe la risposta."""
        level = self.level(model, category, messages)
        alias = self.cfg["livelli"][level]
        emitted = False
        t0 = time.monotonic()
        try:
            self.budget.check()
            capri.attempt(f"claude/{alias}")
            for ev in self.backend.stream(messages, alias):
                if ev[0] == "text":
                    emitted = True
                elif ev[0] == "final":
                    u = getattr(ev[2], "usage", None) or {}
                    self.budget.record(ev[1], u.get("cost_usd", 0.0), u.get("input_tokens", 0),
                                       u.get("output_tokens", 0))
                    log.info("brain categoria=%s livello=%s modello=%s esito=ok costo=%.5f ms=%d",
                             category, level, ev[1], u.get("cost_usd", 0.0), (time.monotonic() - t0) * 1000)
                yield ev
            return
        except BackendError as e:
            log.info("brain categoria=%s livello=%s esito=errore:%s ms=%d", category, level,
                     e.__class__.__name__, (time.monotonic() - t0) * 1000)
            if emitted:
                raise
        # ripiego: modelli gratuiti della categoria (o "generale")
        cats = self.router.cfg["categories"]
        models = cats.get(category or "", cats["generale"])["models"]
        yield from self.router._stream(models, messages, None, category=category)

    def complete(self, messages: list[dict], model: str, category: str | None = None):
        """(modello, messaggio) come Router._complete."""
        used, parts = model, []
        for ev in self.stream(messages, model, category):
            if ev[0] == "final":
                return ev[1], ev[2]
            if ev[0] == "text":
                parts.append(ev[1])
        return used, SimpleNamespace(content="".join(parts), tool_calls=None)
