## PWA løft (videre arbeid)

- Bygg en dedikert PWA-frontend (React/Vite/Next) med manifest, service worker og samme funksjoner som Streamlit-siden: input for tekst/CSV, batchkontroller, fremdrift og tabeller/nedlasting.
- Flytt Python-logikken fra `app.py` (normalisering, batching, OpenAI-kall, parsing) inn i et API-lag (FastAPI/Flask) slik at frontenden kun kaller JSON-endepunkter.
- Implementer `POST /annotate` som tar linjer + innstillinger, instansierer `OpenAI(api_key=user_key)` per request og returnerer output-listen.
- La brukeren legge inn egen OpenAI-nøkkel i frontenden; lagre den kun i sessionStorage/IndexedDB og send den over HTTPS i hver request (aldri logge eller lagre på server).
- Skille feiltyper: 401/429 fra OpenAI, valideringsfeil fra brukerinput, generelle serverfeil – gi tydelige tilbakemeldinger i UI.
- Valgfritt: legg til et lett `/validate-key`-endepunkt som setter opp en minimal testkall slik at brukerne kan sjekke nøkkelen før de kjører store jobber.

