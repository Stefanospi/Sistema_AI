"""Sistema di cache TTL (Time-To-Live) per risposte e metadati.

Fornisce:
- TTLCache: cache thread-safe con scadenza temporale
- response_cache: istanza globale per cache risposte AI (default 5 min)
- model_list_cache: istanza globale per cache lista modelli OpenRouter (default 1 ora)
"""
from __future__ import annotations

import time
import threading
from typing import Any, Optional, Dict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Voce di cache con valore e timestamp di scadenza."""
    value: Any
    expiry: float


class TTLCache:
    """Cache thread-safe con scadenza temporale (TTL).

    Parametri:
        ttl_seconds: durata di vita delle voci in secondi (default 300 = 5 min)
        max_size: numero massimo di voci prima di iniziare l'eviction (default 1000)
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Restituisce il valore associato alla chiave, o None se scaduto/inesistente."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() > entry.expiry:
                del self._store[key]
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Inserisce o aggiorna una voce nella cache."""
        with self._lock:
            if len(self._store) >= self.max_size:
                self._evict_oldest()
            expiry = time.time() + (ttl or self.ttl)
            self._store[key] = CacheEntry(value=value, expiry=expiry)

    def delete(self, key: str) -> bool:
        """Rimuove una voce dalla cache. Restituisce True se esisteva."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> int:
        """Svuota tutta la cache. Restituisce il numero di voci rimosse."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def _evict_oldest(self) -> None:
        """Rimuove la voce più vecchia (per scadenza) per fare spazio."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].expiry)
        del self._store[oldest_key]

    def stats(self) -> dict:
        """Restituisce statistiche sulla cache (size, expired, ttl)."""
        with self._lock:
            now = time.time()
            expired = sum(1 for e in self._store.values() if now > e.expiry)
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "expired_entries": expired,
                "ttl_seconds": self.ttl,
            }


# ---------- Istanze globali (singleton) ----------

response_cache = TTLCache(ttl_seconds=300, max_size=500)   # cache risposte AI: 5 min
model_list_cache = TTLCache(ttl_seconds=3600, max_size=10)  # cache lista modelli: 1 ora