# AI Router — Guida completa

Router personale di modelli AI gratuiti su [OpenRouter](https://openrouter.ai), con:
- 🔀 **Classificazione automatica** della richiesta (per parole chiave, opzionalmente con un LLM)
- 💾 **Memoria delle conversazioni** (SQLite, con riassunto automatico)
- 🛠️ **Agente con strumenti su file** (leggi, modifica, cerca — solo da CLI, con conferma)
- 🌐 **Server compatibile OpenAI** (FastAPI) — usa il router da Continue, Cline o Copilot Chat
- 📸 **Supporto immagini** (screenshot → modelli con visione nella categoria `desktop`)

---

## Indice

1. [Requisiti](#requisiti)
2. [Installazione](#installazione)
3. [Configurazione `.env`](#configurazione-env)
4. [Configurazione `config.yaml`](#configurazione-configyaml)
5. [Avvio — CLI interattiva](#avvio--cli-interattiva)
6. [Avvio — Server compatibile OpenAI](#avvio--server-compatibile-openai)
7. [Configurazione degli editor](#configurazione-degli-editor)
   - [Continue](#continue)
   - [Cline](#cline)
   - [Copilot Chat nativo di VS Code (BYOK)](#copilot-chat-nativo-di-vs-code-byok)
8. [Come aggiornare i modelli](#come-aggiornare-i-modelli)
9. [Come fare lo switch da Copilot al router](#come-fare-lo-switch-da-copilot-al-router)
10. [Limiti noti e rischi](#limiti-noti-e-rischi)

---

## Requisiti

- Python **3.10** o superiore (testato su 3.11)
- Account gratuito su [OpenRouter](https://openrouter.ai) (crea una chiave API gratuita)
- Windows (percorsi e fine riga CRLF gestiti; Linux/macOS funzionano ma non sono il target primario)
- Connessione internet (i modelli girano sui server di OpenRouter, non in locale)

---

## Installazione

```powershell
# 1. Clona o scarica il progetto
cd "C:\Users\spita\Documents\Sistema_AI"

# 2. Crea un ambiente virtuale (consigliato)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Installa le dipendenze
pip install -r requirements.txt

# Per i test aggiungi anche:
pip install -r requirements-dev.txt
```

> **Nota:** se PowerShell blocca l'esecuzione degli script, esegui prima:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Configurazione `.env`

Crea il file `.env` nella cartella del progetto (è già nel `.gitignore`):

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Puoi anche **non** creare il file: la prima volta che avvii `chat.py` o `server.py` ti verrà chiesta la chiave e verrà salvata automaticamente.

**Parametri opzionali:**

```
# Token Bearer per il server (se vuoi proteggere l'endpoint locale)
ROUTER_API_TOKEN=una-stringa-segreta-a-piacere
```

Se `ROUTER_API_TOKEN` non è impostato, il server accetta qualsiasi richiesta (sicuro comunque perché ascolta solo su `127.0.0.1`).

---

## Configurazione `config.yaml`

Il file `config.yaml` controlla tutti i comportamenti del router.

### Sezioni principali

```yaml
fallback_globale: openrouter/free   # ultimo tentativo (sceglie OpenRouter automaticamente)

memoria:
  max_messaggi: 24   # oltre questa soglia la parte vecchia viene riassunta

rate_limit:
  max_tentativi: 2   # riprove sullo stesso modello prima di passare al successivo
  attesa_max_s: 30   # se Retry-After supera questo valore, passa subito al modello successivo

strumenti:           # solo CLI con --agente; ogni modifica chiede conferma
  abilitati: true
  max_passi: 8
  max_read_bytes: 200000
  run_command:
    abilitato: false   # spento di default (PERICOLOSO: attivare solo se necessario)
    consentiti: []     # es. ["git", "python", "pytest"]
    timeout: 60

classificatore_llm:  # classificatore opzionale via LLM (spento di default)
  abilitato: false
  modello: nvidia/nemotron-3.5-lightning:free

log:
  abilitato: true
  file: router.log   # solo metadati (categoria, modello, durata, errori), mai testi o chiavi

server:
  porta: 8000
  reload: false      # true solo in sviluppo
```

### Categorie

Ogni categoria ha:
- `models`: lista **ordinata** di modelli da provare (il primo è il preferito, gli altri sono fallback)
- `keywords`: parole chiave per la classificazione automatica (supporta wildcard con `*`)
- `system`: prompt di sistema inviato al modello
- `sensibile`: se `true`, mostra un avviso sulla privacy quando la categoria viene usata

### Come scegliere la categoria per una richiesta

Il router prova nell'ordine:
1. **Parole chiave**: se la richiesta contiene keyword di una categoria, la usa
2. **Classificatore LLM** (solo se abilitato): usa un piccolo modello per classificare
3. **Generale**: categoria di fallback universale

### Visione (immagini)

```yaml
visione:
  modelli:
    - nvidia/nemotron-3.5-super-vl:free
    - thinkingmachines/inkling-small:free
    - google/gemma-4-26b-a4b-it:free
```

I modelli qui elencati **devono comparire anche nella lista `models` della categoria `desktop`**. Solo quelli nella lista `visione.modelli` vengono usati per le richieste con immagini allegate.

---

## Avvio — CLI interattiva

```powershell
# Avvio base
python chat.py

# Con sessione nominata (memoria separata)
python chat.py --sessione lavoro

# Con agente (strumenti su file, con conferma per ogni modifica)
python chat.py --agente

# Senza streaming (risposta mostrata tutta alla fine)
python chat.py --no-stream

# Lista modelli gratuiti attualmente su OpenRouter
python chat.py --liste-free
```

### Comandi disponibili nella CLI

| Comando | Descrizione |
|---------|-------------|
| `/cat <nome>` | Forza la categoria (es. `/cat coding`) |
| `/auto` | Torna alla classificazione automatica |
| `/img "percorso.png" [domanda]` | Invia uno screenshot (solo categoria `desktop`) |
| `/esci` | Chiude la CLI |

### Esempio di sessione

```
Tu> c'è un bug nella funzione python che legge il file
[coding · nvidia/nemotron-3.5-super-vl:free]
Certo, per diagnosticare il bug ho bisogno di vedere il codice...

Tu> /cat appunti
Categoria forzata: appunti

Tu> /img "C:\screenshot.png" cosa fa questa finestra?
⚠  L'immagine verrà inviata a un provider gratuito che può registrarla.
[desktop · nvidia/nemotron-3.5-super-vl:free]
La finestra mostra...
```

---

## Avvio — Server compatibile OpenAI

Il server espone un'API compatibile con OpenAI che puoi usare da qualsiasi editor o tool che supporta endpoint personalizzati.

### Avvio rapido

```powershell
python server.py
```

Output atteso:
```
Avvio server AI Router su http://127.0.0.1:8000
  Modelli virtuali: GET http://127.0.0.1:8000/v1/models
  Completamenti:   POST http://127.0.0.1:8000/v1/chat/completions
  Autenticazione Bearer: disattivata (server locale, solo 127.0.0.1)
  Premi Ctrl+C per fermare.
```

### Avvio da VS Code (task integrato)

Premi `Ctrl+Shift+P` → **"Tasks: Run Task"** → **"AI Router: avvia server"**

Il server resta in esecuzione in background nel pannello dedicato.

### Endpoint esposti

| Metodo | URL | Descrizione |
|--------|-----|-------------|
| `GET` | `/v1/models` | Lista modelli virtuali disponibili |
| `POST` | `/v1/chat/completions` | Completamenti (con o senza `stream: true`) |

### Modelli virtuali

| ID modello | Descrizione |
|-----------|-------------|
| `router-auto` | Classifica automaticamente la richiesta e sceglie il modello migliore |
| `router-coding` | Usa i modelli della categoria coding |
| `router-finanza` | Usa i modelli della categoria finanza |
| `router-documenti` | Usa i modelli della categoria documenti |
| `router-appunti` | Usa i modelli della categoria appunti |
| `router-desktop` | Usa i modelli della categoria desktop |
| `router-generale` | Usa i modelli della categoria generale |

### Modalità PROXY (predefinita per gli editor)

Quando usi il server da Continue, Cline o Copilot Chat:
- Il server riceve lo **storico completo** dall'editor e lo inoltra al modello scelto
- `messages`, `tools`, `tool_choice` e `stream` vengono passati **invariati** al modello reale
- La **memoria interna del router è disattivata** (l'editor gestisce già il suo storico)
- Gli **strumenti di `tools.py` non sono esposti via HTTP** (sicurezza: solo CLI con conferma)

---

## Configurazione degli editor

### Continue

[Continue](https://www.continue.dev/) è un'estensione open-source per VS Code che supporta qualsiasi provider compatibile OpenAI.

**Installazione:** cerca "Continue" nel marketplace di VS Code e installa l'estensione.

**File di configurazione:** `%USERPROFILE%\.continue\config.yaml`  
(apri con: `Ctrl+Shift+P` → "Continue: Open config.yaml")

Aggiungi uno o più blocchi sotto la chiave `models`:

```yaml
# In %USERPROFILE%\.continue\config.yaml

models:
  # Modello con classificazione automatica (consigliato per uso generale)
  - name: AI Router (auto)
    provider: openai
    model: router-auto
    apiBase: http://127.0.0.1:8000/v1
    apiKey: locale   # valore qualsiasi se ROUTER_API_TOKEN non è impostato
    roles:
      - chat
      - edit
      - apply

  # Modello esplicito per il coding
  - name: AI Router (coding)
    provider: openai
    model: router-coding
    apiBase: http://127.0.0.1:8000/v1
    apiKey: locale
    roles:
      - chat
      - edit
      - apply

  # Autocomplete: usa un modello veloce (scegli quello più adatto alla tua connessione)
  - name: AI Router (appunti)
    provider: openai
    model: router-appunti
    apiBase: http://127.0.0.1:8000/v1
    apiKey: locale
    roles:
      - chat
```

> **Nota:** se hai impostato `ROUTER_API_TOKEN` in `.env`, usa quel valore come `apiKey` anziché `locale`.

> **Nota:** Continue gestisce il proprio storico delle conversazioni, quindi la memoria del router è disabilitata in modalità server — il comportamento è identico a qualsiasi altro provider.

---

### Cline

[Cline](https://cline.bot/) è un'estensione per VS Code che supporta agenti AI con tool calling.

**Installazione:** cerca "Cline" nel marketplace di VS Code.

**Configurazione via UI** (metodo consigliato):

1. Clicca sull'icona di Cline nella barra laterale di VS Code
2. Clicca sull'icona ⚙️ (impostazioni) in alto nel pannello
3. Imposta i campi:
   - **API Provider:** `OpenAI Compatible`
   - **Base URL:** `http://127.0.0.1:8000/v1`
   - **API Key:** `locale` (o il valore di `ROUTER_API_TOKEN` se impostato)
   - **Model ID:** `router-auto` (o il nome virtuale che preferisci, es. `router-coding`)
4. Clicca **Done** / **Save**

**Configurazione via `settings.json`** (alternativa):

```json
{
  "cline.apiProvider": "openai-compatible",
  "cline.openaiCompatibleBaseUrl": "http://127.0.0.1:8000/v1",
  "cline.openaiCompatibleApiKey": "locale",
  "cline.openaiCompatibleModelId": "router-auto"
}
```

> ⚠️ **Avviso tool calling:** Cline usa strumenti nativi (tool calling) intensivamente. I modelli gratuiti su OpenRouter hanno un supporto a tool calling molto variabile e spesso inaffidabile. Il router applica il fallback automatico al modello successivo, ma potrebbe comunque capitare che nessun modello disponibile in quel momento risponda correttamente ai tool call. In quel caso Cline potrebbe bloccarsi o mostrare errori.

---

### Copilot Chat nativo di VS Code (BYOK)

A partire da **VS Code 1.96**, la funzione "Bring Your Own Key" (BYOK) permette di aggiungere modelli personalizzati alla chat nativa di Copilot **anche senza un abbonamento Copilot attivo** (verificato sulla documentazione ufficiale Microsoft).

> ⚠️ **Restrizione enterprise:** Se usi un account **Copilot Business o Enterprise**, l'amministratore della tua organizzazione GitHub deve abilitare esplicitamente la policy "Bring Your Own Language Model Key in VS Code". Senza questo permesso, l'opzione non apparirà.

**Configurazione via UI:**

1. Apri la Chat (`Ctrl+Alt+I`)
2. Clicca sul selettore del modello (in basso nella chat) → **"Manage Language Models"**  
   *(in alternativa: `Ctrl+Shift+P` → "Chat: Manage Language Models")*
3. Clicca **"Add Models..."** → **"Custom Endpoint"**
4. Scegli il gruppo (es. "AI Router locale")
5. Inserisci l'API Key (il valore di `ROUTER_API_TOKEN`, oppure qualsiasi stringa se non impostato)
6. Scegli **API Type: Chat Completions**

VS Code aprirà il file `chatLanguageModels.json` (nella cartella dati utente). Aggiungilo manualmente se preferisci:

```json
[
  {
    "id": "router-auto",
    "name": "AI Router (auto)",
    "vendor": "customendpoint",
    "url": "http://127.0.0.1:8000/v1/chat/completions",
    "apiType": "chat-completions",
    "apiKey": "${input:chat.lm.secret.il-tuo-uuid}",
    "toolCalling": false,
    "vision": false,
    "maxInputTokens": 32000,
    "maxOutputTokens": 4096
  },
  {
    "id": "router-coding",
    "name": "AI Router (coding)",
    "vendor": "customendpoint",
    "url": "http://127.0.0.1:8000/v1/chat/completions",
    "apiType": "chat-completions",
    "apiKey": "${input:chat.lm.secret.il-tuo-uuid}",
    "toolCalling": false,
    "vision": false,
    "maxInputTokens": 32000,
    "maxOutputTokens": 4096
  }
]
```

> **Nota su `toolCalling`:** impostalo a `false` perché il supporto a tool calling dei modelli gratuiti è inaffidabile. VS Code tenterà il tool calling solo con modelli che dichiarano di supportarlo.

> **Nota sull'`apiKey`:** VS Code gestisce le chiavi tramite `SecretStorage` cifrata. Il valore `${input:chat.lm.secret.uuid}` è un riferimento interno che VS Code genera durante la configurazione UI. Se modifichi il file manualmente, usa la UI per impostare la chiave sicura.

---

## Come aggiornare i modelli

I modelli gratuiti su OpenRouter cambiano frequentemente (nuovi modelli, modelli ritirati).

### Controllare quali modelli sono ancora validi

```powershell
# Elenca i modelli gratuiti attualmente disponibili su OpenRouter
python chat.py --liste-free

# Avviando la CLI, vengono segnalati automaticamente i modelli in config.yaml
# che non risultano più su OpenRouter
python chat.py
```

### Aggiornare `config.yaml`

1. Controlla l'elenco completo su `https://openrouter.ai/models?q=:free`
2. Identifica i modelli ritirati (segnalati all'avvio della CLI)
3. Sostituiscili nella lista `models` di ogni categoria
4. Aggiungi i nuovi modelli con visione alla lista `visione.modelli` se applicabile

**Regola:** usa **sempre** l'ID completo nel formato `provider/nome-modello:free` come mostrato da OpenRouter. Non inventare ID.

### Trovare modelli con visione

```powershell
# Filtra i modelli gratuiti che supportano immagini
python -c "
import requests
r = requests.get('https://openrouter.ai/api/v1/models').json()
for m in r['data']:
    if m['id'].endswith(':free') and m.get('architecture', {}).get('input_modalities', []):
        modalities = m['architecture']['input_modalities']
        if 'image' in modalities:
            print(m['id'])
"
```

---

## Come fare lo switch da Copilot al router

### Opzione 1: Affiancamento (consigliata)

Tieni attivo Copilot come "piano B" e usa il router come provider principale negli editor:

- **Continue:** configura il router come primo modello nella lista `models`; Continue ti permette di cambiare modello al volo dal selettore nella chat
- **Cline:** imposta il router come provider, ma puoi cambiarlo in qualsiasi momento dalle impostazioni
- **Copilot Chat nativo:** aggiungi il router come modello aggiuntivo; puoi scegliere qual usare dal selettore in ogni conversazione

In questo modo:
- Se il server del router è spento o tutti i modelli gratuiti sono temporaneamente indisponibili, cambi provider in pochi secondi
- Copilot continua a funzionare per l'autocomplete in linea (suggerimenti durante la digitazione), che il router non sostituisce

### Opzione 2: Switch completo (avanzato)

Se vuoi usare solo il router:
1. Disattiva l'estensione GitHub Copilot dal marketplace (non è necessario disinstallarla)
2. Usa Continue o Cline come unica fonte di AI
3. Riattiva Copilot all'occorrenza

> **Attenzione:** Copilot fornisce l'**autocomplete in linea** (suggerimenti mentre scrivi), che né Continue né Cline replicano allo stesso modo per default. Considera di mantenere Copilot attivo solo per questo, disabilitando la sua chat.

---

## Limiti noti e rischi

Vedi il file [RISCHI.md](./RISCHI.md) per l'analisi completa.

Riepilogo rapido:
- I modelli gratuiti hanno **quote limitate** (~20 req/min) e **rotano frequentemente**
- Il **tool calling** dei modelli gratuiti è spesso inaffidabile
- I provider gratuiti **possono registrare i prompt** — non inviare dati riservati
- Il server risponde solo su `127.0.0.1` — non esporlo su rete pubblica
