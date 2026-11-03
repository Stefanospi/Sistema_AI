"""Backend Claude: lancia la CLI `claude -p` come sottoprocesso e ne legge lo stream.

Usa il login dell'abbonamento (non una chiave API): per questo ANTHROPIC_API_KEY viene tolta
dall'ambiente del figlio, altrimenti la CLI fatturerebbe l'API. Solo testo: nessuno strumento,
nessun file di configurazione utente (--tools "" --safe-mode).

Eventi emessi da `stream` (stessa forma di Router._stream):
  ("start", etichetta_modello), ("text", pezzo), ("final", etichetta_modello, ClaudeMessage)
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger("airouter.backend")

EXE = "claude.exe" if os.name == "nt" else "claude"


class BackendError(RuntimeError):
    """Il backend non ha potuto rispondere (binario assente, timeout, errore della CLI, budget)."""


def find_claude_bin(configured: str = "") -> str:
    """Percorso della CLI: config, CLAUDE_BIN, PATH, estensione VS Code più recente, ~/.local/bin."""
    candidates = [configured, os.getenv("CLAUDE_BIN", ""), shutil.which("claude") or ""]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    ext_dir = Path.home() / ".vscode" / "extensions"
    found = sorted(ext_dir.glob(f"anthropic.claude-code-*/resources/native-binary/{EXE}"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if found:
        return str(found[0])
    local = Path.home() / ".local" / "bin" / EXE
    if local.is_file():
        return str(local)
    raise BackendError("CLI di Claude non trovata: imposta brain.claude.binario in config.yaml "
                       "o la variabile CLAUDE_BIN.")


def _text_of(content) -> str:
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                continue
            parts.append(p.get("text", "") if p.get("type") == "text" else "[allegato omesso]")
        return "\n".join(x for x in parts if x)
    return str(content or "")


def build_prompt(messages: list[dict]) -> tuple[str, str]:
    """(system, prompt) da una lista di messaggi OpenAI. La CLI riceve un solo prompt: lo storico
    precedente viene riassunto come trascrizione, l'ultimo messaggio utente resta in fondo."""
    system = "\n\n".join(_text_of(m.get("content")) for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) <= 1:
        return system, _text_of(turns[0].get("content")) if turns else ""
    labels = {"user": "Utente", "assistant": "Assistente"}
    history = "\n\n".join(f"[{labels[m['role']]}]\n{_text_of(m.get('content'))}" for m in turns[:-1])
    return system, (f"Conversazione finora:\n\n{history}\n\n"
                    f"Ultimo messaggio dell'utente, a cui rispondere:\n{_text_of(turns[-1].get('content'))}")


MAX_IMAGES = 3
MAX_IMAGE_B64 = 6_000_000
DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=\s]+)$")


def last_user_images(messages: list[dict]) -> list[dict]:
    """Blocchi immagine dell'ultimo messaggio utente (data URL base64). Le immagini dei messaggi
    precedenti non si rimandano: costano token a ogni giro."""
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    blocks = []
    for part in (last.get("content") if last and isinstance(last.get("content"), list) else []):
        if not isinstance(part, dict) or part.get("type") == "text":
            continue
        url = (part.get("image_url") or {}).get("url", "") if part.get("type") == "image_url" else ""
        m = DATA_URL.match(url.strip())
        if not m or len(m.group(2)) > MAX_IMAGE_B64:
            raise BackendError("Immagine non valida: servono data URL base64 (png, jpeg, webp, gif) sotto i ~4 MB.")
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": m.group(1),
                                                    "data": re.sub(r"\s+", "", m.group(2))}})
    if len(blocks) > MAX_IMAGES:
        raise BackendError(f"Troppe immagini (massimo {MAX_IMAGES}).")
    return blocks


def _without_images(messages: list[dict]) -> list[dict]:
    """Come messages, ma l'ultimo messaggio utente senza le immagini (viaggiano come blocchi a parte)."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            if isinstance(m.get("content"), list):
                m["content"] = [p for p in m["content"] if isinstance(p, dict) and p.get("type") == "text"]
            break
    return out


class ClaudeCliBackend:
    def __init__(self, cfg: dict | None = None, popen=subprocess.Popen):
        cfg = cfg or {}
        self.binary = cfg.get("binario", "")
        self.timeout = float(cfg.get("timeout_s", 120))
        self.max_budget = cfg.get("max_per_richiesta_usd", 0.05)
        # freno anti-fuga: oltre questo la CLI interrompe con un errore (non tronca), quindi va tenuto alto
        self.max_output = int(cfg.get("max_token_output", 1500))
        self._popen = popen
        # cartella neutra: la CLI non deve leggere CLAUDE.md né file di un progetto qualsiasi
        self.cwd = str(Path(tempfile.gettempdir()))

    def _argv(self, alias: str, system: str, effort: str | None, with_images: bool = False) -> list[str]:
        argv = [find_claude_bin(self.binary), "-p", "--model", alias,
                "--output-format", "stream-json", "--include-partial-messages", "--verbose",
                "--no-session-persistence", "--tools", "", "--safe-mode"]
        if with_images:
            argv += ["--input-format", "stream-json"]  # l'unico modo di passare immagini senza strumenti
        if effort:
            argv += ["--effort", effort]
        if self.max_budget:
            argv += ["--max-budget-usd", str(self.max_budget)]
        if system.strip():
            argv += ["--system-prompt", system]
        return argv

    def stream(self, messages: list[dict], alias: str, effort: str | None = None):
        images = last_user_images(messages)
        system, prompt = build_prompt(_without_images(messages) if images else messages)
        if not prompt.strip():
            raise BackendError("Nessun messaggio utente da inviare.")
        if images:
            payload = json.dumps({"type": "user", "message": {"role": "user",
                                  "content": images + [{"type": "text", "text": prompt}]}}) + "\n"
        else:
            payload = prompt
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        if self.max_output:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.max_output)
        label = f"claude/{alias}"
        try:
            proc = self._popen(self._argv(alias, system, effort, bool(images)), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               cwd=self.cwd, env=env)
        except OSError as e:
            raise BackendError(f"Impossibile avviare la CLI di Claude: {e}")
        timed_out = threading.Event()

        def expire():
            timed_out.set()
            proc.kill()

        timer = threading.Timer(self.timeout, expire)
        timer.start()
        emitted, result, tail, plan = False, None, [], {}
        try:
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
            except OSError:
                pass  # il processo è già morto: l'errore esce dalla lettura dell'output
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    tail = (tail + [line])[-5:]  # righe non JSON: di solito il messaggio d'errore
                    continue
                kind = ev.get("type")
                if kind == "stream_event":
                    inner = ev.get("event") or {}
                    delta = inner.get("delta") or {}
                    if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                        if not emitted:
                            emitted = True
                            yield ("start", label)
                        yield ("text", delta.get("text", ""))
                elif kind == "rate_limit_event":
                    # utilizzo reale del piano (finestre da 5 ore e da 7 giorni, comprende anche Claude Code)
                    windows = (ev.get("rate_limit_info") or {}).get("unifiedWindows") or {}
                    plan = {k: {"utilization": float(w.get("utilization") or 0),
                                "resets_at": float(w.get("resetsAt") or 0)}
                            for k, w in windows.items() if isinstance(w, dict)} or plan
                elif kind == "result":
                    result = ev
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
        if timed_out.is_set():
            raise BackendError(f"La CLI di Claude non ha risposto entro {self.timeout:.0f} secondi.")
        if result is None:
            raise BackendError("La CLI di Claude è terminata senza risposta: " + (" | ".join(tail) or f"codice {proc.returncode}"))
        if result.get("is_error"):
            raise BackendError(str(result.get("result") or result.get("subtype") or "errore della CLI di Claude"))
        text = str(result.get("result") or "")
        if text.startswith("API Error:"):  # es. superato il tetto di token di output
            raise BackendError(text)
        if not emitted:
            yield ("start", label)
            if text:
                yield ("text", text)
        usage = result.get("usage") or {}
        yield ("final", label, SimpleNamespace(
            content=text, tool_calls=None,
            usage={"cost_usd": float(result.get("total_cost_usd") or 0.0),
                   "input_tokens": int(usage.get("input_tokens") or 0)
                   + int(usage.get("cache_creation_input_tokens") or 0)
                   + int(usage.get("cache_read_input_tokens") or 0),
                   "output_tokens": int(usage.get("output_tokens") or 0)},
            plan=plan))
