"""Strumenti dell'agente (solo CLI): confinati nel workspace, con conferma, backup e blocco dei segreti."""
import difflib
import fnmatch
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable

# File che non vanno mai letti né scritti (confronto sul nome, senza maiuscole/minuscole).
SECRET_PATTERNS = [
    ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "*.kdbx", "id_rsa*", "id_ed25519*",
    "id_ecdsa*", "credentials*", "secrets*", ".npmrc", ".pypirc", ".netrc",
]
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ai_backup"}
BACKUP_DIR = ".ai_backup"
MAX_OUTPUT = 20_000

DEFAULTS = {
    "abilitati": True,
    "max_read_bytes": 200_000,
    "max_risultati_ricerca": 50,
    "run_command": {"abilitato": False, "consentiti": [], "timeout": 60},
}


class ToolError(Exception):
    """Errore "normale" di uno strumento: il messaggio viene restituito al modello."""


def _is_secret_name(name: str) -> bool:
    n = name.lower()
    return any(fnmatch.fnmatch(n, p) for p in SECRET_PATTERNS)


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[... troncato, {len(text) - limit} caratteri omessi]"


def tool_schemas(run_command_enabled: bool = False) -> list[dict]:
    def fn(name, desc, props, required):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

    s = {"type": "string"}
    schemas = [
        fn("list_dir", "Elenca file e cartelle di una cartella del workspace.", {"path": s}, []),
        fn("read_file", "Legge un file di testo del workspace.", {"path": s}, ["path"]),
        fn("search_text", "Cerca un testo (o regex) nei file del workspace.",
           {"pattern": s, "path": s, "regex": {"type": "boolean"}}, ["pattern"]),
        fn("edit_file", "Sostituisce UNA occorrenza univoca di old con new in un file (richiede conferma).",
           {"path": s, "old": s, "new": s}, ["path", "old", "new"]),
        fn("write_file", "Crea o sovrascrive un file (richiede conferma).", {"path": s, "content": s},
           ["path", "content"]),
    ]
    if run_command_enabled:
        schemas.append(fn("run_command", "Esegue un comando consentito nel workspace (richiede conferma).",
                          {"command": s}, ["command"]))
    return schemas


class Toolbox:
    def __init__(self, root: Path, cfg: dict | None = None, confirm: Callable[[str], bool] | None = None):
        self.root = Path(root).resolve()
        merged = {**DEFAULTS, **(cfg or {})}
        merged["run_command"] = {**DEFAULTS["run_command"], **(merged.get("run_command") or {})}
        self.cfg = merged
        # senza callback di conferma ogni modifica viene rifiutata
        self.confirm = confirm or (lambda _msg: False)

    @property
    def schemas(self) -> list[dict]:
        return tool_schemas(self.cfg["run_command"]["abilitato"])

    # -- sandbox --
    def resolve(self, path: str | None, allow_secret: bool = False) -> Path:
        raw = (path or ".").strip() or "."
        if "\x00" in raw:
            raise ToolError("Percorso non valido.")
        p = Path(raw)
        if not p.is_absolute() and ":" in raw:  # stream NTFS alternativi, es. file.txt:nascosto
            raise ToolError("Percorso non valido.")
        full = (self.root / p).resolve()  # segue anche i symlink
        if not full.is_relative_to(self.root):
            raise ToolError("Percorso fuori dal workspace: operazione negata.")
        rel = full.relative_to(self.root)
        if ".git" in (x.lower() for x in rel.parts):
            raise ToolError("Accesso a .git negato.")
        if not allow_secret and (_is_secret_name(full.name) or _is_secret_name(Path(raw).name)):
            raise ToolError("File riservato (segreti/chiavi): accesso negato.")
        return full

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."

    def _read_text(self, p: Path) -> str:
        if not p.is_file():
            raise ToolError(f"Non è un file: {self._rel(p)}")
        if p.stat().st_size > self.cfg["max_read_bytes"]:
            raise ToolError("File troppo grande per essere letto.")
        data = p.read_bytes()
        if b"\x00" in data:
            raise ToolError("File binario: non leggibile come testo.")
        try:
            return data.decode("utf-8-sig") if data.startswith(b"\xef\xbb\xbf") else data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError("Il file non è in UTF-8.")

    def _write_text(self, p: Path, text: str, bom: bool = False):
        """Scrittura in binario: nessuna conversione automatica delle fine riga su Windows."""
        data = text.encode("utf-8")
        if bom:
            data = b"\xef\xbb\xbf" + data
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".ai_tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def _backup(self, p: Path) -> str:
        dest_dir = self.root / BACKUP_DIR
        dest_dir.mkdir(exist_ok=True)
        flat = str(p.relative_to(self.root)).replace("\\", "__").replace("/", "__")
        dest = dest_dir / f"{flat}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
        dest.write_bytes(p.read_bytes())
        return dest.name

    # -- strumenti --
    def list_dir(self, path: str = ".") -> str:
        d = self.resolve(path)
        if not d.is_dir():
            raise ToolError(f"Non è una cartella: {self._rel(d)}")
        out = []
        for e in sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if e.name in SKIP_DIRS or _is_secret_name(e.name):
                continue
            out.append(e.name + ("/" if e.is_dir() else ""))
        return "\n".join(out[:500]) or "(cartella vuota)"

    def read_file(self, path: str) -> str:
        return _clip(self._read_text(self.resolve(path)))

    def search_text(self, pattern: str, path: str = ".", regex: bool = False) -> str:
        if not pattern:
            raise ToolError("Pattern vuoto.")
        base = self.resolve(path)
        try:
            rx = re.compile(pattern if regex else re.escape(pattern), re.IGNORECASE)
        except re.error as e:
            raise ToolError(f"Regex non valida: {e}")
        limit = self.cfg["max_risultati_ricerca"]
        hits: list[str] = []
        files = [base] if base.is_file() else self._walk(base)
        for f in files:
            try:
                text = self._read_text(f)
            except (ToolError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{self._rel(f)}:{n}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        return "\n".join(hits) + f"\n[... limite di {limit} risultati raggiunto]"
        return "\n".join(hits) or "Nessun risultato."

    def _walk(self, base: Path):
        for dirpath, dirs, names in os.walk(base):  # followlinks=False di default
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(names):
                f = Path(dirpath) / name
                if _is_secret_name(name):
                    continue
                try:
                    if f.resolve().is_relative_to(self.root):  # ignora i symlink che escono
                        yield f
                except OSError:
                    continue

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self.resolve(path)
        if not old:
            raise ToolError("Il testo da sostituire (old) è vuoto.")
        raw = self._read_text(p)
        bom = p.read_bytes().startswith(b"\xef\xbb\xbf")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        old_n, new_n = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        count = text.count(old_n)
        if count == 0:
            raise ToolError("Testo da sostituire non trovato.")
        if count > 1:
            raise ToolError(f"Il testo compare {count} volte: aggiungi contesto per renderlo univoco.")
        updated = text.replace(old_n, new_n, 1)
        diff = "".join(difflib.unified_diff(text.splitlines(True), updated.splitlines(True),
                                            "prima", "dopo", n=2))
        if not self.confirm(f"Modificare {self._rel(p)}?\n{_clip(diff, 3000)}"):
            return "Modifica rifiutata dall'utente."
        backup = self._backup(p)
        self._write_text(p, updated.replace("\n", "\r\n") if crlf else updated, bom)
        return f"File modificato (backup: {BACKUP_DIR}/{backup})."

    def write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        if p.is_dir():
            raise ToolError("Il percorso è una cartella.")
        exists = p.exists()
        if not self.confirm(f"{'Sovrascrivere' if exists else 'Creare'} {self._rel(p)} ({len(content)} caratteri)?"):
            return "Scrittura rifiutata dall'utente."
        backup = self._backup(p) if exists else None
        self._write_text(p, content)
        return "File scritto" + (f" (backup: {BACKUP_DIR}/{backup})." if backup else ".")

    def run_command(self, command: str) -> str:
        rc = self.cfg["run_command"]
        if not rc["abilitato"]:
            raise ToolError("run_command è disabilitato in config.yaml.")
        try:
            argv = shlex.split(command, posix=(os.name != "nt"))
        except ValueError as e:
            raise ToolError(f"Comando non valido: {e}")
        if not argv:
            raise ToolError("Comando vuoto.")
        exe = argv[0].strip('"')
        if "/" in exe or "\\" in exe or exe.lower() not in {c.lower() for c in rc["consentiti"]}:
            raise ToolError(f"Comando non consentito. Consentiti: {', '.join(rc['consentiti']) or 'nessuno'}.")
        if not self.confirm(f"Eseguire il comando: {command}\n(cartella: {self.root})"):
            return "Comando rifiutato dall'utente."
        try:
            r = subprocess.run([exe, *argv[1:]], cwd=self.root, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=rc["timeout"])
        except subprocess.TimeoutExpired:
            raise ToolError(f"Timeout dopo {rc['timeout']} secondi.")
        except OSError as e:
            raise ToolError(f"Impossibile eseguire il comando: {e}")
        return _clip(f"exit code {r.returncode}\n{r.stdout}{r.stderr}")

    # -- esecuzione da una tool call del modello --
    def execute(self, name: str, arguments: str | dict | None) -> str:
        """Non solleva mai eccezioni: l'errore torna al modello come testo."""
        if not self.cfg["abilitati"]:
            return "Errore: strumenti disabilitati."
        allowed = {s["function"]["name"] for s in self.schemas}
        if name not in allowed:
            return f"Errore: strumento sconosciuto o non abilitato: {name}"
        try:
            if isinstance(arguments, str):
                args = json.loads(arguments) if arguments.strip() else {}
            else:
                args = arguments or {}
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return "Errore: argomenti non validi (serve un oggetto JSON)."
        props = next(s["function"]["parameters"]["properties"] for s in self.schemas
                     if s["function"]["name"] == name)
        required = next(s["function"]["parameters"]["required"] for s in self.schemas
                        if s["function"]["name"] == name)
        missing = [k for k in required if k not in args]
        if missing:
            return f"Errore: argomenti mancanti: {', '.join(missing)}"
        clean = {k: v for k, v in args.items() if k in props}
        for k, v in clean.items():
            if props[k]["type"] == "string" and not isinstance(v, str):
                return f"Errore: l'argomento {k} deve essere una stringa."
        try:
            return getattr(self, name)(**clean)
        except ToolError as e:
            return f"Errore: {e}"
        except OSError as e:
            return f"Errore di accesso al file: {e.__class__.__name__}"


class CachedToolbox(Toolbox):
    """Toolbox con cache per letture file (invalida su modifica del file).

    La cache memorizza (mtime, contenuto) per ogni percorso di file.
    Viene invalidata automaticamente quando il file viene modificato (edit_file/write_file).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._read_cache: dict[str, tuple[float, str]] = {}  # path_str -> (mtime, content)

    def read_file(self, path: str) -> str:
        p = self.resolve(path)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            raise ToolError(f"Non è un file: {self._rel(p)}")

        cached = self._read_cache.get(str(p))
        if cached and cached[0] == mtime:
            return _clip(cached[1])

        content = self._read_text(p)
        self._read_cache[str(p)] = (mtime, content)
        return _clip(content)

    def invalidate_cache(self, path: str | None = None) -> None:
        """Invalida la cache per un percorso specifico o per tutti."""
        if path:
            p = self.resolve(path)
            self._read_cache.pop(str(p), None)
        else:
            self._read_cache.clear()

    def edit_file(self, path: str, old: str, new: str) -> str:
        result = super().edit_file(path, old, new)
        self.invalidate_cache(path)
        return result

    def write_file(self, path: str, content: str) -> str:
        result = super().write_file(path, content)
        self.invalidate_cache(path)
        return result
