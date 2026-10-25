# Rischi e limiti — AI Router

Documento di analisi onesta dei limiti del sistema. Leggilo prima di affidarti al router per lavoro importante.

---

## 1. Tool calling dei modelli gratuiti

**Rischio: ALTO**

Il ciclo agente (strumenti su file, modifica del codice) si basa sul **tool calling nativo** dei modelli — ossia la capacità del modello di rispondere con un JSON strutturato che chiama una funzione specifica anziché scrivere testo libero.

**Problemi osservati con i modelli gratuiti:**

| Problema | Frequenza | Impatto |
|----------|-----------|---------|
| Il modello risponde con testo che _descrive_ il tool call invece di usarlo | Alta | L'agente non esegue nulla |
| Il modello genera JSON malformato negli argomenti | Media | L'esecuzione fallisce con errore controllato |
| Il modello ignora i tool e risponde normalmente | Alta | L'agente si ferma senza completare il compito |
| Il modello risponde 400/404 con `tools` nel body | Media | Il router segna il modello come "senza tool" e passa al successivo |
| Timeout sulla risposta | Bassa | Fallback al modello successivo |

**Mitigazioni implementate:**
- Fallback automatico al modello successivo in caso di 400/404 con tool
- I modelli che rifiutano i tool vengono esclusi per il resto della sessione (`_no_tools`)
- Il numero massimo di passi agente è limitato (`max_passi: 8`)
- Gli strumenti di `tools.py` sono esposti **solo dalla CLI** dove l'utente può controllare ogni passo

**Conclusione:** usa l'agente (`--agente`) solo per operazioni semplici e verifica sempre il risultato. Non usarlo per modifiche critiche in produzione senza revisione manuale.

---

## 2. Privacy e registrazione dei prompt

**Rischio: MEDIO-ALTO per dati sensibili**

I modelli gratuiti su OpenRouter sono serviti da provider terzi (NVIDIA, Google, Cohere, ecc.). I tuoi prompt vengono inviati ai loro server.

**Cosa possono fare i provider:**
- Registrare i prompt per il training dei modelli (dipende dalla policy di ogni provider)
- Analizzare il traffico per motivi di sicurezza/abuso
- I dati trasmessi potrebbero essere soggetti alle leggi del paese del provider

**Mitigazioni implementate:**
- Avviso automatico in CLI per le categorie contrassegnate come `sensibile: true` (finanza, documenti)
- Avviso esplicito prima dell'invio di immagini nella categoria desktop
- I log locali non registrano mai il contenuto delle richieste o delle risposte

**Cosa NON fare:**
- Non inviare dati personali di clienti, dipendenti o pazienti
- Non inviare credenziali, chiavi API, password (il router blocca i file `.env`, `.key`, ecc. ma non i testi che li contengono)
- Non inviare segreti aziendali, NDA, brevetti non ancora depositati
- Non inviare documenti soggetti a GDPR o normative sanitarie (HIPAA)

---

## 3. Quote e disponibilità

**Rischio: MEDIO**

I modelli gratuiti hanno **limiti di frequenza** (circa 20 richieste/minuto per account) e limiti mensili di token che variano per modello.

**Problemi possibili:**
- Errori 429 (troppe richieste): il router attende e riprova, poi passa al modello successivo
- Tutti i modelli di una categoria potrebbero essere temporaneamente indisponibili simultaneamente
- I modelli gratuiti vengono ritirati senza preavviso (cambiano le policy di OpenRouter o dei provider)
- Performance degradata nelle ore di punta (i modelli gratuiti hanno priorità bassa)

**Mitigazioni implementate:**
- Retry con backoff esponenziale (rispetta `Retry-After` se presente)
- Fallback progressivo: prova i modelli in ordine, poi il `fallback_globale`
- Rilevamento automatico dei modelli ritirati all'avvio della CLI

**Consigli pratici:**
- Mantieni più modelli nella lista di ogni categoria (almeno 2-3)
- Controlla periodicamente la lista con `python chat.py --liste-free`
- Per uso intensivo, considera un piano a pagamento su OpenRouter (prezzi molto contenuti)

---

## 4. Qualità delle risposte

**Rischio: MEDIO**

I modelli gratuiti sono versioni ridotte o modelli meno recenti. Rispetto a GPT-4o, Claude 3.5 Sonnet o Gemini 1.5 Pro:

- **Ragionamento complesso:** significativamente inferiore
- **Coding avanzato:** sufficiente per task semplici, problematico per architetture complesse
- **Coerenza nelle sessioni lunghe:** la memoria con riassunto mitiga ma non elimina il problema
- **Lingue:** l'italiano è generalmente supportato ma la qualità è variabile

**Caso d'uso ideale:** domande singole, riassunti, snippet di codice, spiegazioni concettuali.  
**Caso d'uso problematico:** refactoring di grandi codebase, debugging di problemi complessi, analisi finanziaria precisa.

---

## 5. Sicurezza del server

**Rischio: BASSO (se usato correttamente)**

Il server ascolta **solo su `127.0.0.1`** — non è raggiungibile dalla rete locale o da internet. L'host non può essere cambiato via `config.yaml` (hardcoded per sicurezza).

**Rischi residui:**
- Qualsiasi processo in esecuzione sulla tua macchina può fare richieste al server (inclusi malware)
- Se imposti `ROUTER_API_TOKEN`, tienilo riservato come qualsiasi altra password
- Con tool calling attivo in Cline, Cline stesso potrebbe inviare tool call al router — ma il router non espone `tools.py` via HTTP, quindi il rischio è limitato

**Non fare mai:**
- Esporre il server su `0.0.0.0` o su un IP di rete
- Usare il server su una macchina condivisa senza autenticazione Bearer
- Disabilitare le conferme di `tools.py` anche nella CLI

---

## 6. Dipendenza da OpenRouter

**Rischio: BASSO-MEDIO**

Il progetto dipende interamente da [OpenRouter](https://openrouter.ai) come aggregatore. Se OpenRouter:
- Cambia le sue policy gratuite (probabile nel lungo periodo)
- Ha un'interruzione del servizio
- Rimuove modelli gratuiti in blocco

...il router smette di funzionare finché non aggiorni i modelli o cambi provider.

**Mitigazioni possibili (non implementate):**
- Aggiungere un secondo provider (es. Groq API free tier) come alternativa
- Configurare modelli locali con Ollama come ultima risorsa (richiede hardware adeguato)

---

## 7. Compatibilità con VS Code / Copilot

**Funzionalità VS Code nativa (BYOK) — Rischio: MEDIO**

La funzione "Bring Your Own Key" di Copilot Chat (vedi `chatLanguageModels.json`) è disponibile da VS Code 1.96 ma:
- **Non disponibile con piani Copilot Business/Enterprise** senza approvazione amministrativa
- L'interfaccia e i campi del file di configurazione potrebbero cambiare in futuro (feature relativamente nuova)
- `toolCalling: false` è necessario: i modelli gratuiti non supportano tool calling in modo affidabile e abilitarlo potrebbe causare comportamenti imprevisti nella chat nativa

**Continue e Cline — Rischio: BASSO**

Queste estensioni usano un'API stabile e documentata. Il formato `config.yaml` di Continue potrebbe cambiare con versioni major dell'estensione: controlla la documentazione ufficiale dopo ogni aggiornamento importante.

---

## Riepilogo

| Area | Rischio | Note |
|------|---------|-------|
| Tool calling agente | 🔴 Alto | Usare con cautela, verificare sempre |
| Privacy dati sensibili | 🟠 Medio-Alto | Non inviare dati riservati |
| Quote e disponibilità | 🟡 Medio | Tenere la lista modelli aggiornata |
| Qualità risposte | 🟡 Medio | Adeguata per task semplici |
| Sicurezza server locale | 🟢 Basso | Solo su 127.0.0.1, conforme by design |
| Dipendenza OpenRouter | 🟡 Medio | Monitorare le policy gratuite |
| Compatibilità editor | 🟢 Basso | API stabile, testata |
