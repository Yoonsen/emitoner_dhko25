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
- `APP_PASSWORD` (valgfri; hvis satt må brukeren logge inn før appen lastes).
- Valgfritt: alle øvrige secrets er fjernet; appen krever ikke passord/gate.

### Kjøring lokalt
1. Installer uv: `pip install uv` (eller bruk eksisterende).
2. Fra repo-roten:
   ```
   OPENAI_API_KEY=sk-... uv run streamlit run app.py
   ```
3. Appen starter på `http://localhost:8501/`.

### Nøkkelfunksjoner
- Dynamisk liste over kategorifelter: legg til/fjern felt, hver med navn + kommaseparerte verdier og modus (`Unik` vs `Liste`, sistnevnte støtter opptil tre verdier).
- Alle verdilister får automatisk `uten-relevans`; prompten minner modellen om å bruke den når ingenting passer.
- Tilpassbar target-markering (default `<b>…</b>`); instruksen forklarer at fragmentene alltid har strukturen `A<start>X<slutt>B`.
- Geotagging-modus: aktiver geodata og velg felt som historisk navn, moderne navn, land og koordinater – feltene kobles inn i prompten og eksporteres som egne kolonner.
- Instruksjonsfeltet + teknisk prompt oppdateres live basert på feltene og markørene.
- Kjøring kan gjøres på sample (med token-estimat) eller hele datasettet, med progressbar og robust JSON-parsing.
- Resultater lastes ned som JSONL/CSV, der analysekolonnene alltid kommer før originale kildekolonner.

### Dokumentasjon
- `architecture.md` beskriver dataflyt, state, promptkonstruksjon, eksport og anbefalt API/React-oppdeling. All videreutvikling (f.eks. migrering til JavaScript/React) bør følge denne spesifikasjonen fremfor å lese direkte fra `app.py`.

### Videre arbeid (PWA-plan)
- Bygg eget API-lag (FastAPI/Flask) som eksponerer `POST /annotate`, gjenbruker logikken spesifisert i `architecture.md`.
- Ny frontend (React/Vite/Next) som henter `manifest.json`, service worker, installasjon.
- Bruker legger inn egen OpenAI-nøkkel klient-side; backend instansierer `OpenAI(api_key=...)` per request og lagrer aldri nøkkelen.
- Valgfritt `/validate-key`-endepunkt og tydelig feilhåndtering (401/429 vs inputfeil).

