import os
import subprocess
import sys

import pytest

from tools import Toolbox


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


def tb(ws, answer=True, **cfg):
    asked = []

    def confirm(m):
        asked.append(m)
        return answer

    t = Toolbox(ws, cfg, confirm=confirm)
    t.asked = asked
    return t


def test_sandbox_blocca_traversal_e_assoluti(ws, tmp_path):
    (tmp_path / "fuori.txt").write_text("x")
    t = tb(ws)
    assert "fuori dal workspace" in t.execute("read_file", {"path": "../fuori.txt"})
    assert "fuori dal workspace" in t.execute("read_file", {"path": str(tmp_path / "fuori.txt")})
    assert "Errore" in t.execute("write_file", {"path": "../nuovo.txt", "content": "x"})
    assert not (tmp_path / "nuovo.txt").exists()
    assert t.asked == []


def test_sandbox_blocca_symlink_che_esce(ws, tmp_path):
    target = tmp_path / "esterno"
    target.mkdir()
    (target / "s.txt").write_text("segreto")
    link = ws / "link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink non permessi (su Windows servono privilegi)")
    t = tb(ws)
    assert "fuori dal workspace" in t.execute("read_file", {"path": "link/s.txt"})
    assert "Nessun risultato" in t.execute("search_text", {"pattern": "segreto"})


def test_junction_windows_non_esce(ws, tmp_path):
    if os.name != "nt":
        pytest.skip("solo Windows")
    target = tmp_path / "esterno"
    target.mkdir()
    (target / "s.txt").write_text("segreto")
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(ws / "j"), str(target)], capture_output=True)
    if r.returncode != 0:
        pytest.skip("junction non creabile")
    assert "fuori dal workspace" in tb(ws).execute("read_file", {"path": "j/s.txt"})


def test_segreti_bloccati(ws):
    for name in (".env", ".ENV", ".env.local", "server.pem", "id_rsa"):
        (ws / name).write_text("TOP=1")
    t = tb(ws)
    for name in (".env", ".ENV", ".env.local", "server.pem", "id_rsa"):
        assert "riservato" in t.execute("read_file", {"path": name})
        assert "riservato" in t.execute("write_file", {"path": name, "content": "x"})
    assert ".env" not in t.execute("list_dir", {})
    assert "Nessun risultato" in t.execute("search_text", {"pattern": "TOP"})
    assert "riservato" in t.execute("edit_file", {"path": ".env", "old": "TOP", "new": "X"})


def test_stream_alternativi_ntfs_rifiutati(ws):
    (ws / "a.txt").write_text("x")
    assert "non valido" in tb(ws).execute("read_file", {"path": "a.txt:nascosto"})


def test_edit_univoco(ws):
    (ws / "a.txt").write_text("uno\ndue\nuno\n", encoding="utf-8")
    t = tb(ws)
    assert "2 volte" in t.execute("edit_file", {"path": "a.txt", "old": "uno", "new": "X"})
    assert "non trovato" in t.execute("edit_file", {"path": "a.txt", "old": "zzz", "new": "X"})
    assert "vuoto" in t.execute("edit_file", {"path": "a.txt", "old": "", "new": "X"})
    assert (ws / "a.txt").read_text(encoding="utf-8") == "uno\ndue\nuno\n"
    assert "modificato" in t.execute("edit_file", {"path": "a.txt", "old": "due", "new": "TRE"})
    assert (ws / "a.txt").read_text(encoding="utf-8") == "uno\nTRE\nuno\n"
    assert any((ws / ".ai_backup").iterdir())


def test_edit_preserva_crlf_e_bom(ws):
    (ws / "w.txt").write_bytes(b"\xef\xbb\xbfriga1\r\nriga2\r\nriga3\r\n")
    t = tb(ws)
    # il modello manda "\n": deve comunque trovare e scrivere CRLF
    assert "modificato" in t.execute("edit_file", {"path": "w.txt", "old": "riga1\nriga2", "new": "A\nB\nC"})
    assert (ws / "w.txt").read_bytes() == b"\xef\xbb\xbfA\r\nB\r\nC\r\nriga3\r\n"


def test_conferma_rifiutata_non_modifica(ws):
    (ws / "a.txt").write_text("uno", encoding="utf-8")
    t = tb(ws, answer=False)
    assert "rifiutata" in t.execute("edit_file", {"path": "a.txt", "old": "uno", "new": "due"})
    assert "rifiutata" in t.execute("write_file", {"path": "b.txt", "content": "x"})
    assert (ws / "a.txt").read_text() == "uno" and not (ws / "b.txt").exists()
    assert not (ws / ".ai_backup").exists()


def test_senza_callback_si_rifiuta_tutto(ws):
    assert "rifiutata" in Toolbox(ws).execute("write_file", {"path": "b.txt", "content": "x"})
    assert not (ws / "b.txt").exists()


def test_write_file_backup_su_sovrascrittura(ws):
    (ws / "a.txt").write_text("vecchio", encoding="utf-8")
    t = tb(ws)
    assert "backup" in t.execute("write_file", {"path": "a.txt", "content": "nuovo"})
    assert (ws / "a.txt").read_text() == "nuovo"
    assert next((ws / ".ai_backup").iterdir()).read_text() == "vecchio"
    assert "scritto" in t.execute("write_file", {"path": "sub/n.txt", "content": "x"})


def test_argomenti_malformati(ws):
    t = tb(ws)
    assert "non validi" in t.execute("read_file", "{rotto")
    assert "non validi" in t.execute("read_file", "[1,2]")
    assert "mancanti" in t.execute("read_file", "{}")
    assert "stringa" in t.execute("read_file", {"path": 5})
    assert "sconosciuto" in t.execute("rm_rf", {})
    assert "Errore" in t.execute("search_text", {"pattern": "(", "regex": True})
    assert "Non è un file" in t.execute("read_file", {"path": "."})


def test_file_binario_e_non_utf8(ws):
    (ws / "b.bin").write_bytes(b"\x00\x01")
    (ws / "l.txt").write_bytes("è".encode("latin-1"))
    t = tb(ws)
    assert "binario" in t.execute("read_file", {"path": "b.bin"})
    assert "UTF-8" in t.execute("read_file", {"path": "l.txt"})


def test_search_e_list_dir(ws):
    (ws / "src").mkdir()
    (ws / "src" / "m.py").write_text("def Ciao():\n    pass\n")
    (ws / ".git").mkdir()
    (ws / ".git" / "x").write_text("ciao")
    t = tb(ws)
    assert "src/" in t.execute("list_dir", {}) and ".git" not in t.execute("list_dir", {})
    assert t.execute("search_text", {"pattern": "ciao"}).replace("\\", "/") == "src/m.py:1: def Ciao():"
    assert "Accesso a .git negato" in t.execute("read_file", {"path": ".git/x"})


def test_run_command_spento_di_default(ws):
    t = tb(ws)
    assert "sconosciuto" in t.execute("run_command", {"command": "python -V"})
    assert "run_command" not in str(t.schemas)


def test_run_command_allowlist_e_conferma(ws):
    t = tb(ws, run_command={"abilitato": True, "consentiti": ["python"], "timeout": 20})
    assert "non consentito" in t.execute("run_command", {"command": "calc"})
    assert "non consentito" in t.execute("run_command", {"command": "./python -V"})
    assert t.asked == []
    out = t.execute("run_command", {"command": f'"{sys.executable}" -V'})
    assert "non consentito" in out  # il percorso completo non è nell'elenco
    assert "exit code 0" in t.execute("run_command", {"command": "python -c pass"})
    assert len(t.asked) == 1
    t2 = tb(ws, answer=False, run_command={"abilitato": True, "consentiti": ["python"]})
    assert "rifiutato" in t2.execute("run_command", {"command": "python -c pass"})
