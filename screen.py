"""Domande sullo schermo ("dove salvo il file?"): prompt e lettura della risposta.

Capri manda la finestra attiva come elenco di controlli (UI Automation: id, tipo, nome) e, se l'app non
li espone, anche uno screenshot di quella sola finestra. Claude risponde in JSON indicando UN controllo
dell'elenco (o un punto dello screenshot); qui si traduce in un bersaglio che Capri può indicare.
"""
from __future__ import annotations

import json
import re

MAX_ELEMENTS = 300
MAX_NAME = 80

SYSTEM = (
    "Sei Capri, l'assistente sul desktop dell'utente. Ti arrivano l'elenco dei controlli visibili della "
    "finestra attiva (id|tipo|nome) ed eventualmente uno screenshot di quella finestra. Rispondi alla domanda "
    "in italiano, breve e concreto (massimo tre frasi; passi numerati solo se servono). Se la domanda è dove "
    "trovare o cosa premere, indica UN solo controllo.\n"
    'Rispondi SOLO con un oggetto JSON: {"risposta": "...", "elemento": <id dell\'elenco oppure null>, '
    '"punto": [x, y] oppure null}.\n'
    '"elemento" quando uno dei controlli elencati è quello giusto (non inventare id). "punto" solo se nell\'elenco '
    "non c'è e hai lo screenshot: pixel dello screenshot, origine in alto a sinistra, centro dell'elemento. "
    "Se non sai dove sia, usa null e dillo nella risposta. Se la domanda riguarda quali programmi o finestre sono "
    "aperti, rispondi con l'elenco delle finestre e usa null per \"elemento\" e \"punto\"."
)
MAX_WINDOWS = 60


def _clean(value, limit: int = MAX_NAME) -> str:
    s = re.sub(r"[\s|]+", " ", str(value or "")).strip()
    return s if len(s) <= limit else s[:limit] + "…"


def normalize_elements(elements: list[dict]) -> list[dict]:
    """Solo i campi utili, con id numerico e rettangolo intero; elementi malformati scartati."""
    out = []
    for e in elements or []:
        if len(out) >= MAX_ELEMENTS:
            break
        try:
            out.append({"i": int(e["i"]), "t": _clean(e.get("t"), 24), "n": _clean(e.get("n")),
                        "a": _clean(e.get("a"), 40), "e": bool(e.get("e", True)),
                        "rect": [int(e["x"]), int(e["y"]), int(e["w"]), int(e["h"])]})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def normalize_windows(windows: list[dict]) -> list[dict]:
    out = []
    for w in windows or []:
        if len(out) >= MAX_WINDOWS:
            break
        if not isinstance(w, dict) or not (w.get("process") or w.get("title")):
            continue
        out.append({"p": _clean(w.get("process"), 40), "t": _clean(w.get("title"), 100), "fg": bool(w.get("fg"))})
    return out


def build_messages(question: str, window: dict, elements: list[dict], image: str | None = None,
                   image_size: list[int] | None = None, windows: list[dict] | None = None) -> list[dict]:
    els = normalize_elements(elements)
    wins = normalize_windows(windows)
    lines = []
    if wins:
        lines.append("Finestre aperte (programma — titolo):")
        lines += [f"{w['p']} — {w['t']}" + (" (in primo piano)" if w["fg"] else "") for w in wins]
    if window and (window.get("process") or window.get("title")):
        lines.append(f"Finestra attiva: {_clean(window.get('process'), 40)} — {_clean(window.get('title'), 120)}")
        if els:
            lines.append("Controlli (id|tipo|nome):")
            for e in els:
                label = e["n"] or (f"[{e['a']}]" if e["a"] else "")
                lines.append(f"{e['i']}|{e['t']}|{label}" + ("" if e["e"] else "|disattivato"))
        else:
            lines.append("Nessun controllo leggibile: affidati allo screenshot.")
    if image and image_size and len(image_size) == 2:
        lines.append(f"Lo screenshot è di {int(image_size[0])}x{int(image_size[1])} pixel.")
    lines.append(f"Domanda: {question.strip()}")
    text = "\n".join(lines)
    content = [{"type": "text", "text": text}]
    if image:
        content.insert(0, {"type": "image_url", "image_url": {"url": image}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def _extract_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    for candidate in (raw, *re.findall(r"\{.*\}", raw, flags=re.DOTALL)[:1]):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_answer(raw: str, elements: list[dict], image_size: list[int] | None = None) -> dict:
    """{"text": risposta, "target": None | {"source": "uia", id, name, rect} | {"source": "image", x, y}}.
    Se Claude non ha risposto in JSON, il testo grezzo è la risposta e non c'è bersaglio."""
    data = _extract_json(raw)
    if data is None:
        return {"text": (raw or "").strip(), "target": None}
    text = str(data.get("risposta") or "").strip() or (raw or "").strip()
    by_id = {e["i"]: e for e in normalize_elements(elements)}
    wanted = data.get("elemento")
    if isinstance(wanted, (int, float)) and not isinstance(wanted, bool) and int(wanted) in by_id:
        e = by_id[int(wanted)]
        return {"text": text, "target": {"source": "uia", "id": e["i"], "name": e["n"], "rect": e["rect"]}}
    point = data.get("punto")
    if (isinstance(point, list) and len(point) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point)
            and image_size and len(image_size) == 2
            and 0 <= point[0] <= image_size[0] and 0 <= point[1] <= image_size[1]):
        return {"text": text, "target": {"source": "image", "x": float(point[0]), "y": float(point[1])}}
    return {"text": text, "target": None}
