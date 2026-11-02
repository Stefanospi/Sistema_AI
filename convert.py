"""Conversione di file con librerie Python: nessun modello, nessun token.

  convert("C:/Users/me/Documents/relazione.docx", "pdf")  ->  C:/Users/me/Documents/relazione.pdf

Il file di partenza non viene mai toccato; il risultato va accanto all'originale e non sovrascrive mai
niente (se esiste già diventa "nome (1).pdf"). Si lavora solo dentro le cartelle consentite (di
default la tua cartella utente) e i file con nome da segreto (.env, chiavi...) sono rifiutati.
Le librerie si importano solo quando servono: se una manca, l'errore dice quale installare.
"""
from __future__ import annotations

import csv
import html as html_lib
import importlib
import io
import json
import re
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape

from tools import _is_secret_name

MAX_MB_DEFAULT = 50
ALIASES = {"jpeg": "jpg", "tif": "tiff", "yml": "yaml", "htm": "html", "markdown": "md", "text": "txt"}
IMAGES = {"png", "jpg", "webp", "bmp", "gif", "tiff", "ico"}
PIL_FORMAT = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP", "bmp": "BMP", "gif": "GIF", "tiff": "TIFF", "ico": "ICO"}


class ConversionError(Exception):
    """Errore "normale": il messaggio è pensato per essere mostrato all'utente."""


def ext(name: str) -> str:
    e = name.lower().lstrip(".").rsplit(".", 1)[-1] if "." in name else name.lower().lstrip(".")
    return ALIASES.get(e, e)


def _lib(module: str, pip: str | None = None):
    try:
        return importlib.import_module(module)
    except ImportError:
        raise ConversionError(f"Manca la libreria {pip or module}: pip install {pip or module}")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


# ---------------------------------------------------------------- immagini

def _flatten(im):
    """RGB senza trasparenza (JPEG, BMP e PDF non la supportano): sfondo bianco."""
    Image = _lib("PIL.Image", "Pillow")
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return im.convert("RGB")


def image_to_image(src: Path, dst: Path) -> None:
    Image = _lib("PIL.Image", "Pillow")
    fmt = PIL_FORMAT[ext(dst.name)]
    with Image.open(src) as im:
        if fmt in ("JPEG", "BMP"):
            im = _flatten(im)
        elif fmt == "ICO":
            side = max(im.size)
            im.save(dst, fmt, sizes=[(s, s) for s in (16, 32, 48, 64, 128, 256) if s <= max(side, 16)])
            return
        elif fmt == "GIF" and im.mode not in ("P", "L"):
            im = im.convert("RGBA") if "A" in im.mode else im.convert("RGB")
        im.save(dst, fmt)


def image_to_pdf(src: Path, dst: Path) -> None:
    Image = _lib("PIL.Image", "Pillow")
    with Image.open(src) as im:
        _flatten(im).save(dst, "PDF", resolution=150.0)


# ---------------------------------------------------------------- testo

def _html_doc(body: str, title: str) -> str:
    return ("<!doctype html>\n<html lang=\"it\"><head><meta charset=\"utf-8\">"
            f"<title>{html_lib.escape(title)}</title></head>\n<body>\n{body}\n</body></html>\n")


def md_to_html_body(text: str) -> str:
    markdown = _lib("markdown")
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def txt_to_md(src: Path, dst: Path) -> None:
    dst.write_text(read_text(src), encoding="utf-8")


def md_to_txt(src: Path, dst: Path) -> None:
    dst.write_text(html_to_text(md_to_html_body(read_text(src))), encoding="utf-8")


def md_to_html(src: Path, dst: Path) -> None:
    dst.write_text(_html_doc(md_to_html_body(read_text(src)), src.stem), encoding="utf-8")


def txt_to_html(src: Path, dst: Path) -> None:
    paras = [p for p in re.split(r"\n\s*\n", read_text(src).replace("\r\n", "\n")) if p.strip()]
    body = "\n".join("<p>" + html_lib.escape(p.strip()).replace("\n", "<br>\n") + "</p>" for p in paras)
    dst.write_text(_html_doc(body, src.stem), encoding="utf-8")


def html_to_text(markup: str) -> str:
    bs4 = _lib("bs4", "beautifulsoup4")
    soup = bs4.BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = [ln.rstrip() for ln in soup.get_text("\n").splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def html_to_txt(src: Path, dst: Path) -> None:
    dst.write_text(html_to_text(read_text(src)), encoding="utf-8")


def _blocks(text: str):
    """Righe di un testo/markdown semplice come (tipo, livello, testo): heading, bullet, number, para."""
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            yield "heading", len(m.group(1)), m.group(2)
        elif re.match(r"^[-*•]\s+", s):
            yield "bullet", 0, re.sub(r"^[-*•]\s+", "", s)
        elif re.match(r"^\d+[.)]\s+", s):
            yield "number", 0, re.sub(r"^\d+[.)]\s+", "", s)
        else:
            yield "para", 0, s


def text_to_docx(src: Path, dst: Path) -> None:
    docx = _lib("docx", "python-docx")
    doc = docx.Document()
    for kind, level, text in _blocks(read_text(src)):
        text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
        if kind == "heading":
            doc.add_heading(text, level=min(level, 4))
        elif kind == "bullet":
            doc.add_paragraph(text, style="List Bullet")
        elif kind == "number":
            doc.add_paragraph(text, style="List Number")
        else:
            doc.add_paragraph(text)
    doc.save(dst)


def docx_to_markdown(src: Path) -> str:
    docx = _lib("docx", "python-docx")
    doc = docx.Document(src)
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    blocks: list[tuple[str, str]] = []  # (tipo, testo) nell'ordine del documento
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            m = re.match(r"Heading (\d)", style)
            if style == "Title":
                blocks.append(("heading", f"# {text}"))
            elif m:
                blocks.append(("heading", "#" * min(int(m.group(1)), 6) + f" {text}"))
            elif style.startswith("List Bullet"):
                blocks.append(("bullet", f"- {text}"))
            elif style.startswith("List Number"):
                blocks.append(("number", f"1. {text}"))
            else:
                blocks.append(("para", text))
        elif tag == "tbl":
            rows = [[c.text.strip().replace("\n", " ") for c in r.cells] for r in Table(child, doc).rows]
            if rows:
                lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                blocks.append(("table", "\n".join(lines)))
    out: list[str] = []
    prev = None
    for kind, text in blocks:
        # le voci di una lista stanno attaccate; ogni altro blocco è separato da una riga vuota
        if prev is not None and not (kind == prev and kind in ("bullet", "number")):
            out.append("")
        out.append(text)
        prev = kind
    return "\n".join(out).strip() + "\n"


def docx_to_md(src: Path, dst: Path) -> None:
    dst.write_text(docx_to_markdown(src), encoding="utf-8")


def docx_to_txt(src: Path, dst: Path) -> None:
    dst.write_text(html_to_text(md_to_html_body(docx_to_markdown(src))), encoding="utf-8")


def docx_to_html(src: Path, dst: Path) -> None:
    dst.write_text(_html_doc(md_to_html_body(docx_to_markdown(src)), src.stem), encoding="utf-8")


def _inline(text: str) -> str:
    """Grassetto/corsivo/codice del markdown come tag di reportlab (dopo l'escape)."""
    s = escape(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<i>\1</i>", s)
    return re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", s)


def markdown_to_pdf(markdown_text: str, dst: Path, title: str) -> None:
    platypus = _lib("reportlab.platypus", "reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    styles = getSampleStyleSheet()
    flow = []
    for kind, level, text in _blocks(markdown_text):
        if kind == "heading":
            flow.append(platypus.Paragraph(_inline(text), styles[f"Heading{min(level, 4)}"]))
        elif kind == "bullet":
            flow.append(platypus.Paragraph(_inline(text), styles["BodyText"], bulletText="•"))
        elif kind == "number":
            flow.append(platypus.Paragraph(_inline(text), styles["BodyText"], bulletText="–"))
        else:
            flow.append(platypus.Paragraph(_inline(text), styles["BodyText"]))
        flow.append(platypus.Spacer(1, 3))
    if not flow:
        flow.append(platypus.Paragraph(" ", styles["BodyText"]))
    platypus.SimpleDocTemplate(str(dst), pagesize=A4, title=title, leftMargin=2 * cm, rightMargin=2 * cm,
                               topMargin=2 * cm, bottomMargin=2 * cm).build(flow)


def text_to_pdf(src: Path, dst: Path) -> None:
    markdown_to_pdf(read_text(src), dst, src.stem)


def docx_to_pdf(src: Path, dst: Path) -> None:
    markdown_to_pdf(docx_to_markdown(src), dst, src.stem)


def pdf_to_txt(src: Path, dst: Path) -> None:
    pypdf = _lib("pypdf")
    reader = pypdf.PdfReader(str(src))
    if reader.is_encrypted:
        raise ConversionError("Il PDF è protetto da password.")
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ConversionError("Nessun testo estraibile: forse è una scansione (serve un OCR).")
    dst.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------- dati

def _num(cell: str):
    s = cell.strip()
    if re.fullmatch(r"-?\d+", s) and not (len(s) > 1 and s.startswith("0")):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return cell


def read_csv_rows(path: Path) -> list[list[str]]:
    text = read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return [row for row in csv.reader(io.StringIO(text), dialect)]


def read_xlsx_rows(path: Path) -> list[list]:
    openpyxl = _lib("openpyxl")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        return [["" if v is None else v for v in row] for row in ws.iter_rows(values_only=True)]
    finally:
        wb.close()


def read_json_rows(path: Path) -> list[list]:
    data = json.loads(read_text(path))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ConversionError("Il JSON deve essere una lista di oggetti (o di liste) per diventare una tabella.")
    if all(isinstance(r, dict) for r in data):
        keys: list[str] = []
        for r in data:
            keys += [k for k in r if k not in keys]
        return [keys] + [[_cell(r.get(k)) for k in keys] for r in data]
    if all(isinstance(r, list) for r in data):
        return [[_cell(v) for v in r] for r in data]
    raise ConversionError("Il JSON deve essere una lista di oggetti (o di liste) per diventare una tabella.")


def _cell(v):
    return "" if v is None else v if isinstance(v, (int, float, str, bool)) else json.dumps(v, ensure_ascii=False)


def write_csv_rows(rows: list[list], dst: Path) -> None:
    with dst.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel apre gli accenti giusti
        csv.writer(f).writerows(rows)


def write_xlsx_rows(rows: list[list], dst: Path) -> None:
    openpyxl = _lib("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(list(row))
    wb.save(dst)


def write_json_rows(rows: list[list], dst: Path) -> None:
    if not rows:
        dst.write_text("[]\n", encoding="utf-8")
        return
    header, body = [str(h) for h in rows[0]], rows[1:]
    data = [dict(zip(header, r)) for r in body]
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _rows_from(src: Path) -> list[list]:
    kind = ext(src.name)
    if kind == "csv":
        return [[_num(c) for c in row] for row in read_csv_rows(src)]
    if kind == "xlsx":
        return read_xlsx_rows(src)
    return read_json_rows(src)


def table_converter(writer: Callable[[list[list], Path], None]):
    def run(src: Path, dst: Path) -> None:
        writer(_rows_from(src), dst)
    return run


def json_to_yaml(src: Path, dst: Path) -> None:
    yaml = _lib("yaml", "pyyaml")
    dst.write_text(yaml.safe_dump(json.loads(read_text(src)), allow_unicode=True, sort_keys=False), encoding="utf-8")


def yaml_to_json(src: Path, dst: Path) -> None:
    yaml = _lib("yaml", "pyyaml")
    dst.write_text(json.dumps(yaml.safe_load(read_text(src)), ensure_ascii=False, indent=2, default=str) + "\n",
                   encoding="utf-8")


# ---------------------------------------------------------------- registro

CONVERTERS: dict[tuple[str, str], Callable[[Path, Path], None]] = {}


def _register():
    for a in IMAGES:
        for b in IMAGES:
            if a != b:
                CONVERTERS[(a, b)] = image_to_image
        CONVERTERS[(a, "pdf")] = image_to_pdf
    CONVERTERS.update({
        ("txt", "md"): txt_to_md, ("md", "txt"): md_to_txt,
        ("md", "html"): md_to_html, ("txt", "html"): txt_to_html, ("html", "txt"): html_to_txt,
        ("txt", "docx"): text_to_docx, ("md", "docx"): text_to_docx,
        ("docx", "txt"): docx_to_txt, ("docx", "md"): docx_to_md, ("docx", "html"): docx_to_html,
        ("txt", "pdf"): text_to_pdf, ("md", "pdf"): text_to_pdf, ("docx", "pdf"): docx_to_pdf,
        ("pdf", "txt"): pdf_to_txt,
        ("csv", "xlsx"): table_converter(write_xlsx_rows), ("csv", "json"): table_converter(write_json_rows),
        ("xlsx", "csv"): table_converter(write_csv_rows), ("xlsx", "json"): table_converter(write_json_rows),
        ("json", "csv"): table_converter(write_csv_rows), ("json", "xlsx"): table_converter(write_xlsx_rows),
        ("json", "yaml"): json_to_yaml, ("yaml", "json"): yaml_to_json,
    })


_register()


def supported() -> dict[str, list[str]]:
    """Formato di partenza -> formati in cui si può convertire."""
    out: dict[str, list[str]] = {}
    for (a, b) in CONVERTERS:
        out.setdefault(a, []).append(b)
    return {a: sorted(bs) for a, bs in sorted(out.items())}


# ---------------------------------------------------------------- uso

def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        cand = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not cand.exists():
            return cand
    raise ConversionError("Troppi file con lo stesso nome nella cartella.")


def convert(src: str | Path, to: str, roots: list[str | Path] | None = None, max_mb: float = MAX_MB_DEFAULT) -> Path:
    path = Path(str(src).strip().strip('"')).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError:
        raise ConversionError(f"File non trovato: {src}")
    if not path.is_file():
        raise ConversionError("Il percorso non è un file.")
    allowed = [Path(r).expanduser().resolve() for r in (roots or [Path.home()])]
    if not any(path.is_relative_to(r) for r in allowed):
        raise ConversionError("File fuori dalle cartelle consentite (conversione.cartelle_consentite in config.yaml).")
    if _is_secret_name(path.name):
        raise ConversionError("File riservato (segreti/chiavi): conversione rifiutata.")
    if path.stat().st_size > max_mb * 1024 * 1024:
        raise ConversionError(f"File troppo grande (massimo {max_mb:g} MB).")

    a, b = ext(path.name), ext(to)
    fn = CONVERTERS.get((a, b))
    if fn is None:
        options = supported().get(a)
        hint = f" Da .{a} posso arrivare a: {', '.join(options)}." if options else f" Il formato .{a} non è supportato."
        raise ConversionError(f"Non so convertire .{a} in .{b}." + hint)

    dst = _unique(path.with_name(f"{path.stem}.{b}"))
    try:
        fn(path, dst)
    except ConversionError:
        dst.unlink(missing_ok=True)
        raise
    except Exception as e:  # file rovinato, formato inatteso...
        dst.unlink(missing_ok=True)
        raise ConversionError(f"Conversione fallita ({e.__class__.__name__}): {e}")
    return dst
