"""Conversione file: librerie vere su file piccoli creati nei test, nessuna rete."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import convert  # noqa: E402
from convert import ConversionError  # noqa: E402

PIL = pytest.importorskip("PIL.Image")
pytest.importorskip("docx")
pytest.importorskip("openpyxl")
pytest.importorskip("pypdf")
pytest.importorskip("reportlab")
pytest.importorskip("markdown")
pytest.importorskip("bs4")


def run(path, to, tmp_path):
    return convert.convert(path, to, roots=[tmp_path])


# ---------- immagini ----------

def make_png(path, mode="RGBA"):
    from PIL import Image
    Image.new(mode, (40, 30), (255, 0, 0, 128) if mode == "RGBA" else (255, 0, 0)).save(path)
    return path


def test_png_con_trasparenza_a_jpg_e_webp(tmp_path):
    src = make_png(tmp_path / "foto.png")
    jpg = run(src, "jpg", tmp_path)
    assert jpg.name == "foto.jpg" and jpg.stat().st_size > 0
    from PIL import Image
    with Image.open(jpg) as im:
        assert im.format == "JPEG" and im.size == (40, 30)
    assert Image.open(run(src, ".WEBP", tmp_path)).format == "WEBP"


def test_immagine_a_pdf_e_ico(tmp_path):
    src = make_png(tmp_path / "logo.png")
    pdf = run(src, "pdf", tmp_path)
    assert pdf.read_bytes().startswith(b"%PDF")
    from PIL import Image
    assert Image.open(run(src, "ico", tmp_path)).format == "ICO"


def test_non_sovrascrive_mai(tmp_path):
    src = make_png(tmp_path / "a.png")
    first = run(src, "jpg", tmp_path)
    second = run(src, "jpg", tmp_path)
    third = run(src, "jpg", tmp_path)
    assert [first.name, second.name, third.name] == ["a.jpg", "a (1).jpg", "a (2).jpg"]
    assert src.exists()


# ---------- testo e documenti ----------

MD = "# Titolo\n\nUn **paragrafo** con àccènti è €.\n\n- uno\n- due\n\n1. primo\n2. secondo\n"


def test_md_a_docx_e_ritorno(tmp_path):
    src = tmp_path / "nota.md"
    src.write_text(MD, encoding="utf-8")
    docx_path = run(src, "docx", tmp_path)
    back = run(docx_path, "md", tmp_path)
    text = back.read_text(encoding="utf-8")
    assert "# Titolo" in text and "paragrafo" in text and "àccènti è €" in text
    assert "- uno" in text and "1. primo" in text


def test_docx_a_txt_html_pdf(tmp_path):
    src = tmp_path / "nota.md"
    src.write_text(MD, encoding="utf-8")
    d = run(src, "docx", tmp_path)
    assert "paragrafo" in run(d, "txt", tmp_path).read_text(encoding="utf-8")
    html = run(d, "html", tmp_path).read_text(encoding="utf-8")
    assert "<h1>Titolo</h1>" in html and "<li>uno</li>" in html and "charset" in html
    assert run(d, "pdf", tmp_path).read_bytes().startswith(b"%PDF")


def test_docx_con_tabella(tmp_path):
    import docx
    doc = docx.Document()
    doc.add_paragraph("Prima")
    t = doc.add_table(rows=2, cols=2)
    for i, v in enumerate(["a", "b", "1", "2"]):
        t.cell(i // 2, i % 2).text = v
    path = tmp_path / "t.docx"
    doc.save(path)
    text = run(path, "md", tmp_path).read_text(encoding="utf-8")
    assert "| a | b |" in text and "| 1 | 2 |" in text


def test_txt_e_md_a_pdf_e_pdf_a_txt(tmp_path):
    src = tmp_path / "relazione.txt"
    src.write_text("Prima riga\n\nSeconda riga con accento: è\n", encoding="cp1252")  # codifica Windows
    pdf = run(src, "pdf", tmp_path)
    text = run(pdf, "txt", tmp_path).read_text(encoding="utf-8")
    assert "Prima riga" in text and "Seconda riga" in text


def test_html_a_txt_toglie_script_e_stile(tmp_path):
    src = tmp_path / "p.html"
    src.write_text("<html><style>p{}</style><body><h1>Ciao</h1><script>x()</script><p>Mondo</p></body></html>",
                   encoding="utf-8")
    text = run(src, "txt", tmp_path).read_text(encoding="utf-8")
    assert "Ciao" in text and "Mondo" in text and "x()" not in text and "p{}" not in text


def test_pdf_senza_testo(tmp_path):
    from PIL import Image
    pdf = tmp_path / "scansione.pdf"
    Image.new("RGB", (50, 50), "white").save(pdf, "PDF")
    with pytest.raises(ConversionError, match="OCR"):
        run(pdf, "txt", tmp_path)
    assert not (tmp_path / "scansione.txt").exists()   # niente file a metà


# ---------- dati ----------

def test_csv_xlsx_json_in_giro(tmp_path):
    src = tmp_path / "dati.csv"
    src.write_text("nome;età;prezzo\nMario;30;1.5\nLucia;25;2\n", encoding="utf-8")
    xlsx = run(src, "xlsx", tmp_path)
    import openpyxl
    ws = openpyxl.load_workbook(xlsx).active
    assert [c.value for c in ws[1]] == ["nome", "età", "prezzo"]
    assert [c.value for c in ws[2]] == ["Mario", 30, 1.5]          # numeri veri, separatore ; riconosciuto
    js = run(xlsx, "json", tmp_path)
    assert json.loads(js.read_text(encoding="utf-8"))[1] == {"nome": "Lucia", "età": 25, "prezzo": 2}
    back = run(js, "csv", tmp_path)
    assert back.read_text(encoding="utf-8-sig").splitlines()[0] == "nome,età,prezzo"
    assert run(js, "xlsx", tmp_path).exists()


def test_json_annidato_in_csv(tmp_path):
    src = tmp_path / "x.json"
    src.write_text(json.dumps([{"a": 1, "b": {"c": 2}}, {"a": 3, "z": None}]), encoding="utf-8")
    rows = run(src, "csv", tmp_path).read_text(encoding="utf-8-sig").splitlines()
    assert rows[0] == "a,b,z" and rows[1].startswith("1,") and "c" in rows[1]


def test_json_non_tabellare(tmp_path):
    src = tmp_path / "x.json"
    src.write_text('"solo una stringa"', encoding="utf-8")
    with pytest.raises(ConversionError, match="lista"):
        run(src, "csv", tmp_path)


def test_json_yaml_e_ritorno(tmp_path):
    src = tmp_path / "c.json"
    src.write_text('{"nome": "è", "lista": [1, 2]}', encoding="utf-8")
    y = run(src, "yml", tmp_path)
    assert y.suffix == ".yaml" and "nome: è" in y.read_text(encoding="utf-8")
    assert json.loads(run(y, "json", tmp_path).read_text(encoding="utf-8")) == {"nome": "è", "lista": [1, 2]}


# ---------- sicurezza e errori ----------

def test_coppia_non_supportata_elenca_le_alternative(tmp_path):
    src = tmp_path / "a.csv"
    src.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ConversionError, match="Da .csv posso arrivare a: json, xlsx"):
        run(src, "mp3", tmp_path)


def test_formato_sconosciuto(tmp_path):
    src = tmp_path / "a.xyz"
    src.write_text("x")
    with pytest.raises(ConversionError, match="non è supportato"):
        run(src, "pdf", tmp_path)


def test_fuori_dalle_cartelle_consentite(tmp_path):
    outside = tmp_path / "fuori"
    outside.mkdir()
    allowed = tmp_path / "ok"
    allowed.mkdir()
    src = make_png(outside / "a.png")
    with pytest.raises(ConversionError, match="fuori dalle cartelle"):
        convert.convert(src, "jpg", roots=[allowed])


def test_percorso_con_dotdot_non_scappa(tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    make_png(tmp_path / "segreto.png")
    with pytest.raises(ConversionError, match="fuori dalle cartelle"):
        convert.convert(str(allowed / ".." / "segreto.png"), "jpg", roots=[allowed])


def test_file_segreti_rifiutati(tmp_path):
    for name in (".env", "id_rsa", "server.pem"):
        (tmp_path / name).write_text("TOP=1")
    with pytest.raises(ConversionError, match="riservato"):
        run(tmp_path / ".env", "txt", tmp_path)


def test_limite_dimensione_e_file_mancante(tmp_path):
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * 2048)
    with pytest.raises(ConversionError, match="troppo grande"):
        convert.convert(big, "md", roots=[tmp_path], max_mb=0.001)
    with pytest.raises(ConversionError, match="non trovato"):
        run(tmp_path / "non_esiste.png", "jpg", tmp_path)
    with pytest.raises(ConversionError, match="non è un file"):
        run(tmp_path, "jpg", tmp_path)


def test_file_rovinato_non_lascia_residui(tmp_path):
    bad = tmp_path / "rotto.png"
    bad.write_bytes(b"non sono un png")
    with pytest.raises(ConversionError, match="fallita"):
        run(bad, "jpg", tmp_path)
    assert not (tmp_path / "rotto.jpg").exists()


def test_libreria_mancante_dice_quale_installare(tmp_path, monkeypatch):
    real = convert.importlib.import_module

    def fake(name, *a, **k):
        if name == "openpyxl":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(convert.importlib, "import_module", fake)
    src = tmp_path / "a.csv"
    src.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ConversionError, match="pip install openpyxl"):
        run(src, "xlsx", tmp_path)


def test_supported_elenca_le_coppie():
    s = convert.supported()
    assert "pdf" in s["docx"] and "jpg" in s["png"] and s["pdf"] == ["txt"]
    assert "jpeg" not in s       # alias normalizzato
