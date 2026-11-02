"""Cervello: sceglie livello e modello Claude per ogni richiesta, contiene i consumi (budget in dollari
e quota reale del piano) e riduce i token inviati.

Con `modelli_gratuiti: false` (predefinito) niente esce verso OpenRouter: ogni richiesta senza strumenti
passa da Claude, quindi anche le categorie sensibili. Se Claude non può rispondere (budget, quota, CLI)
la richiesta fallisce con un messaggio chiaro invece di ripiegare su provider che registrano i prompt.

Modelli virtuali del server: brain-auto (livello deciso dalla policy), brain-basso, brain-medio, brain-alto.
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
WEEK_S = 7 * 86_400
PLAN_WINDOWS = {"five_hour": ("finestra_5h_max", "da 5 ore"), "seven_day": ("finestra_7g_max", "settimanale")}

DEFAULTS = {
    "abilitato": True,
    "modelli_gratuiti": False,          # False = mai verso OpenRouter; True = ripiego ammesso (non per le categorie sensibili)
    "claude": {"binario": "", "timeout_s": 120, "max_per_richiesta_usd": 0.02, "max_token_output": 1500},
    "livelli": {"basso": "haiku", "medio": "sonnet", "alto": "opus"},
    "effort": {"basso": "low", "medio": "medium", "alto": "high"},
    "livello_default": "basso",
    "per_categoria": {},
    "salita": [],
    "limiti": {"max_messaggi_storico": 6, "max_caratteri_prompt": 16000},
    "stile": ("Rispondi in italiano, in modo diretto e conciso: niente premesse né riassunto finale. "
              "Entra nei dettagli solo se richiesto."),
    "budget": {"giornaliero_usd": 0.5, "settimanale_usd": 2.0},
    "quota_piano": {"finestra_5h_max": 0.5, "finestra_7g_max": 0.4},
}


class BudgetExceeded(BackendError):
    """Tetto di spesa raggiunto."""


class QuotaExceeded(BackendError):
    """La quota reale del piano ha superato la soglia riservata a questo assistente."""


class Budget:
    """Spesa in dollari a prezzo di listino (come la riporta la CLI) e ultime letture della quota del piano.
    Con l'abbonamento la spesa non è un addebito: è un contatore di consumo e un freno."""

    def __init__(self, path: Path | None = None, daily_usd: float = 0, weekly_usd: float = 0, clock=time.time):
        self.path = path or BASE_DIR / "brain_usage.db"
        self.daily_usd = float(daily_usd or 0)
        self.weekly_usd = float(weekly_usd or 0)
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
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS plan(
                     window TEXT PRIMARY KEY, utilization REAL NOT NULL, resets_at REAL NOT NULL,
                     updated REAL NOT NULL)""")
            self._db.commit()
        return self._db

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(self._clock()))

    def _spent(self, where: str, arg) -> float:
        with self._lock:
            row = self._conn().execute(f"SELECT COALESCE(SUM(cost),0) FROM usage WHERE {where}", (arg,)).fetchone()
        return float(row[0])

    def spent_today(self) -> float:
        return self._spent("day=?", self._today())

    def spent_week(self) -> float:
        return self._spent("ts>=?", self._clock() - WEEK_S)

    def check(self) -> None:
        if self.daily_usd and self.spent_today() >= self.daily_usd:
            raise BudgetExceeded(f"Budget giornaliero raggiunto ({self.daily_usd:.2f} $): riprova domani.")
        if self.weekly_usd and self.spent_week() >= self.weekly_usd:
            raise BudgetExceeded(f"Budget settimanale raggiunto ({self.weekly_usd:.2f} $).")

    def record(self, model: str, cost: float, tokens_in: int = 0, tokens_out: int = 0) -> None:
        with self._lock:
            db = self._conn()
            db.execute("INSERT INTO usage(day, ts, model, cost, tokens_in, tokens_out) VALUES(?,?,?,?,?,?)",
                       (self._today(), self._clock(), model, cost, tokens_in, tokens_out))
            db.commit()

    # -- quota reale del piano (letta dagli eventi della CLI: include anche Claude Code) --
    def record_plan(self, plan: dict) -> None:
        if not plan:
            return
        with self._lock:
            db = self._conn()
            for window, w in plan.items():
                db.execute("INSERT OR REPLACE INTO plan(window, utilization, resets_at, updated) VALUES(?,?,?,?)",
                           (window, w["utilization"], w["resets_at"], self._clock()))
            db.commit()

    def plan_state(self) -> dict:
        """Ultime letture ancora valide: una finestra già scaduta conta come libera."""
        with self._lock:
            rows = self._conn().execute("SELECT window, utilization, resets_at FROM plan").fetchall()
        now = self._clock()
        return {w: {"utilization": u, "resets_at": r} for w, u, r in rows if r > now}

    def check_plan(self, limits: dict) -> None:
        state = self.plan_state()
        for window, (key, label) in PLAN_WINDOWS.items():
            limit, w = float(limits.get(key) or 0), state.get(window)
            if limit and w and w["utilization"] >= limit:
                until = time.strftime("%d/%m %H:%M", time.localtime(w["resets_at"]))
                raise QuotaExceeded(
                    f"Quota del piano {label} al {w['utilization']:.0%} (soglia {limit:.0%}): "
                    f"sospeso fino al {until} per lasciare spazio a Claude Code.")

    def status(self) -> dict:
        def left(limit, spent):
            return round(max(limit - spent, 0.0), 5) if limit else None

        day, week = self.spent_today(), self.spent_week()
        return {"spent_today_usd": round(day, 5), "daily_usd": self.daily_usd, "remaining_today_usd": left(self.daily_usd, day),
                "spent_week_usd": round(week, 5), "weekly_usd": self.weekly_usd, "remaining_week_usd": left(self.weekly_usd, week),
                "plan": self.plan_state()}


def _merge(base: dict, over: dict | None) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def _text_len(content) -> int:
    if isinstance(content, list):
        return sum(len(str(p.get("text", ""))) for p in content if isinstance(p, dict))
    return len(str(content or ""))


def _last_user_text(messages: list[dict]) -> str:
    last = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    if isinstance(last, list):
        last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
    return str(last or "")


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.4)
    return text[:head] + f"\n[... {len(text) - head - tail} caratteri omessi per limitare i token ...]\n" + text[-tail:]


class Brain:
    def __init__(self, router, backend=None, budget: Budget | None = None, cfg: dict | None = None):
        self.router = router
        self.cfg = _merge(DEFAULTS, cfg if cfg is not None else router.cfg.get("brain"))
        self.backend = backend or ClaudeCliBackend(self.cfg["claude"])
        self.budget = budget or Budget(daily_usd=self.cfg["budget"]["giornaliero_usd"],
                                       weekly_usd=self.cfg["budget"]["settimanale_usd"])
        # la CLI e le sessioni del server passano da qui; i riassunti usano il livello più economico
        router.brain = self
        router.summarizer = self._summarize

    def _summarize(self, prompt: str) -> str:
        return self.complete([{"role": "user", "content": prompt}], "brain-basso", "appunti")[1].content or ""

    # -- quali richieste gestisce --
    def handles(self, model: str, tools) -> bool:
        """brain-* sempre; con i modelli gratuiti spenti anche router-* e gli ID OpenRouter.
        Con gli strumenti no: la CLI qui gira solo in testo."""
        if not self.cfg["abilitato"] or tools:
            return False
        return model.startswith(PREFIX) or not self.cfg["modelli_gratuiti"]

    def virtual_models(self) -> list[str]:
        return [AUTO] + [PREFIX + lv for lv in LEVELS] if self.cfg["abilitato"] else []

    def allow_free(self, category: str | None) -> bool:
        sensitive = bool((self.router.cfg["categories"].get(category or "") or {}).get("sensibile"))
        return bool(self.cfg["modelli_gratuiti"]) and not sensitive

    # -- policy: livello e modello --
    def level(self, model: str, category: str | None, messages: list[dict]) -> str:
        forced = model[len(PREFIX):] if model.startswith(PREFIX) else ""
        if forced in LEVELS:
            return forced
        level = self.cfg["per_categoria"].get(category) or self.cfg["livello_default"]
        if level not in LEVELS:
            level = "basso"
        text = _last_user_text(messages).lower()
        if any(k.lower() in text for k in self.cfg["salita"]):
            level = LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]
        return level

    # -- meno token --
    def prepare(self, messages: list[dict]) -> list[dict]:
        """Aggiunge l'istruzione di concisione e taglia lo storico: ultimi N messaggi, e comunque
        entro il limite di caratteri (si scartano prima i più vecchi; l'ultimo non viene mai scartato)."""
        lim = self.cfg["limiti"]
        system = [m for m in messages if m.get("role") == "system"]
        turns = [m for m in messages if m.get("role") != "system"][-int(lim["max_messaggi_storico"]):]
        budget_chars = int(lim["max_caratteri_prompt"])
        while len(turns) > 1 and sum(_text_len(m.get("content")) for m in turns) > budget_chars:
            turns.pop(0)
        if turns and isinstance(turns[-1].get("content"), str) and len(turns[-1]["content"]) > budget_chars:
            turns[-1] = {**turns[-1], "content": _clip_middle(turns[-1]["content"], budget_chars)}
        if turns and turns[0].get("role") == "assistant" and len(turns) > 1:
            turns.pop(0)  # un taglio non deve lasciare una risposta orfana in testa
        style = str(self.cfg["stile"]).strip()
        base = "\n\n".join(str(m.get("content") or "").strip() for m in system if m.get("content"))
        return [{"role": "system", "content": (base + "\n\n" + style).strip()}] + turns

    def status(self) -> dict:
        return {**self.budget.status(), "livelli": self.cfg["livelli"], "livello_default": self.cfg["livello_default"],
                "modelli_gratuiti": self.cfg["modelli_gratuiti"], "soglie_piano": self.cfg["quota_piano"]}

    # -- esecuzione --
    def stream(self, messages: list[dict], model: str, category: str | None = None):
        """Stessi eventi di Router._stream. Un errore di Claude prima del primo testo ripiega sui modelli
        gratuiti solo se consentiti (mai per le categorie sensibili); altrimenti viene sollevato."""
        level = self.level(model, category, messages)
        alias = self.cfg["livelli"][level]
        emitted, failure = False, None
        t0 = time.monotonic()
        try:
            self.budget.check()
            self.budget.check_plan(self.cfg["quota_piano"])
            prepared = self.prepare(messages)
            capri.attempt(f"claude/{alias}")
            for ev in self.backend.stream(prepared, alias, effort=self.cfg["effort"].get(level)):
                if ev[0] == "text":
                    emitted = True
                elif ev[0] == "final":
                    u = getattr(ev[2], "usage", None) or {}
                    self.budget.record(ev[1], u.get("cost_usd", 0.0), u.get("input_tokens", 0), u.get("output_tokens", 0))
                    self.budget.record_plan(getattr(ev[2], "plan", None))
                    log.info("brain categoria=%s livello=%s modello=%s esito=ok costo=%.5f tok_in=%d tok_out=%d ms=%d",
                             category, level, ev[1], u.get("cost_usd", 0.0), u.get("input_tokens", 0),
                             u.get("output_tokens", 0), (time.monotonic() - t0) * 1000)
                yield ev
            return
        except BackendError as e:
            failure = e
            log.info("brain categoria=%s livello=%s esito=errore:%s ms=%d", category, level,
                     e.__class__.__name__, (time.monotonic() - t0) * 1000)
        if emitted or not self.allow_free(category):
            raise failure
        cats = self.router.cfg["categories"]
        models = cats.get(category or "", cats["generale"])["models"]
        yield from self.router._stream(models, messages, None, category=category)

    def complete(self, messages: list[dict], model: str, category: str | None = None):
        """(modello, messaggio) come Router._complete."""
        parts = []
        for ev in self.stream(messages, model, category):
            if ev[0] == "final":
                return ev[1], ev[2]
            if ev[0] == "text":
                parts.append(ev[1])
        return model, SimpleNamespace(content="".join(parts), tool_calls=None)
