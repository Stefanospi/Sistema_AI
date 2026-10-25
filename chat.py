"""CLI per il router AI. Uso: python chat.py [--sessione nome] [--liste-free]"""
import argparse
import os
import shlex
from pathlib import Path

import requests

from router import Router, all_model_ids
from tools import Toolbox


def ask_confirm(message: str) -> bool:
    """Conferma nel terminale: qualsiasi risposta diversa da s/si viene trattata come no."""
    print(f"\n--- CONFERMA RICHIESTA ---\n{message}")
    try:
        return input("Confermi? [s/N] ").strip().lower() in ("s", "si", "sì", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessione", default="default", help="nome della conversazione (memoria separata)")
    ap.add_argument("--liste-free", action="store_true", help="mostra i modelli :free attualmente su OpenRouter")
    ap.add_argument("--agente", action="store_true", help="abilita gli strumenti su file (con conferma) nella cartella corrente")
    ap.add_argument("--no-stream", action="store_true", help="disattiva lo streaming (risposta mostrata alla fine)")
    args = ap.parse_args()

    if args.liste_free:
        for m in sorted(i for i in all_model_ids() if i.endswith(":free")):
            print(m)
        return

    router = Router()
    cats = list(router.cfg["categories"])

    try:
        missing = router.check_config()
        if missing:
            print("Attenzione, questi modelli in config.yaml non risultano più su OpenRouter:")
            for m in missing:
                print("  -", m)
            print("Verranno saltati grazie al fallback. Aggiorna config.yaml quando puoi.\n")
    except requests.RequestException:
        print("(Impossibile controllare l'elenco modelli, procedo comunque.)\n")

    print(f"Sessione: {args.sessione} | categorie: {', '.join(cats)}")
    toolbox = None
    if args.agente:
        toolbox = Toolbox(Path.cwd(), router.cfg.get("strumenti"), confirm=ask_confirm)
        print(f"Agente attivo nella cartella: {toolbox.root}")
    print("Comandi: /cat <nome> forza una categoria, /auto torna al riconoscimento automatico,")
    print('         /img "percorso.png" [domanda] invia uno screenshot (categoria desktop), /esci\n')

    forced = None
    warned = set()
    while True:
        try:
            text = input("Tu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/esci", "/exit"):
            break
        if text == "/auto":
            forced = None
            print("Riconoscimento automatico attivo.")
            continue
        if text == "/cat" or text.startswith("/cat "):
            name = text[4:].strip()
            if name in cats:
                forced = name
                print(f"Categoria forzata: {name}")
            else:
                print("Categorie valide:", ", ".join(cats))
            continue

        images = None
        if text == "/img" or text.startswith("/img "):
            try:
                parts = shlex.split(text[4:], posix=(os.name != "nt"))
            except ValueError:
                parts = []
            if not parts:
                print('Uso: /img "percorso\\screenshot.png" [domanda]')
                continue
            images = [parts[0].strip('"')]
            text = " ".join(parts[1:]) or "Descrivi questa schermata."
            forced_now = "desktop"
            print("⚠  L'immagine verrà inviata a un provider gratuito che può registrarla.")
        else:
            forced_now = None

        category = forced_now or forced or router.classify(text)
        if router.cfg["categories"][category].get("sensibile") and category not in warned:
            print("⚠  I modelli gratuiti possono registrare i prompt: non inviare dati riservati o di clienti.")
            warned.add(category)

        shown = []

        def on_model(m, category=category):
            if not shown:
                print(f"\n[{category} · {m}]")
                shown.append(m)

        def on_text(chunk):
            print(chunk, end="", flush=True)

        try:
            if args.no_stream:
                category, model, answer = router.ask(args.sessione, text, category, toolbox, images=images)
                print(f"\n[{category} · {model}]\n{answer}\n")
            else:
                router.ask(args.sessione, text, category, toolbox, on_text, on_model, images)
                print("\n")
        except (RuntimeError, ValueError) as e:
            print(f"\nErrore: {e}\n")
        except KeyboardInterrupt:
            print("\nInterrotto.\n")


if __name__ == "__main__":
    main()
