## Luminoner manifest

### Formål
- Streamlit-app for batchannotering av tekstfragmenter mot OpenAI Chat Completions.
- Brukerdefinerte kategori-felt (kolonner) med egne verdilister, automatisk catch-all `uten-relevans`.
- Resultater returneres som tabeller, grafer og kan lastes ned som JSONL/CSV.

### Avhengigheter
- Python 3.11+ (prosjekt bruker `uv` + `pyproject.toml`).
- `streamlit`, `openai`, `pandas`, m.m. (se `pyproject.toml`).

### Miljøvariabler/secrets
- `OPENAI_API_KEY` (kreves, enten i `.streamlit/secrets.toml` eller env).
- Valgfritt: alle øvrige secrets er fjernet; appen krever ikke passord/gate.

### Kjøring lokalt
1. Installer uv: `pip install uv` (eller bruk eksisterende).
2. Fra repo-roten:
   ```
   OPENAI_API_KEY=sk-... uv run streamlit run app.py
   ```
3. Appen starter på `http://localhost:8501/`.

### Nøkkelfunksjoner
- Dynamisk liste over kategorifelter: legg til/fjern felt, hver med navn + kommaseparerte verdier.
- Alle verdilister får automatisk `uten-relevans`; prompten minner modellen om å bruke den når ingenting passer.
- Instruksjonsfeltet + teknisk prompt oppdateres live basert på feltene.
- Batchkjøring med testmodus, progressbar og eksport (JSONL/CSV).

### Videre arbeid (PWA-plan)
- Bygg eget API-lag (FastAPI/Flask) som eksponerer `POST /annotate`, gjenbruker dagens Pythonlogikk.
- Ny frontend (React/Vite/Next) som henter `manifest.json`, service worker, installasjon.
- Bruker legger inn egen OpenAI-nøkkel klient-side; backend instansierer `OpenAI(api_key=...)` per request og lagrer aldri nøkkelen.
- Valgfritt `/validate-key`-endepunkt og tydelig feilhåndtering (401/429 vs inputfeil).

