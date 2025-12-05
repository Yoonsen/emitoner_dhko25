## Luminoner – Arkitektur og dataflyt

Dette dokumentet er kilden som beskriver hvordan Streamlit-versjonen fungerer. Når vi bygger en React/JS-frontend eller egen backend bør vi følge spesifikasjonen her fremfor å lese direkte fra `app.py`.

---

### 1. Hovedkomponenter
- **UI/State**: Streamlit holder `session_state` for kategorioppsett, target-markører, sampling og sist brukte kolonneheadere.
- **OpenAI-klient**: Ett `OpenAI(api_key=...)`-objekt brukes for alle batchkall i en kjøring.
- **Resultathåndtering**: Alle modellresponser normaliseres, merges med kildekolonner og eksporteres som JSONL + CSV.

---

### 2. Brukeroppsett
1. **Kategorier**
   - Hvert felt har `label`, `values` (kommaseparert vokabular) og `mode`.
   - `mode = "unique"` → eksakt én verdi fra vokabularet.
   - `mode = "list"` → inntil tre verdier fra samme vokabular (returneres som liste i JSON, pipe-separert i CSV).
   - `uten-relevans` legges automatisk til alle vokabularer og brukes som fallback.

2. **Target-markører**
   - Brukeren angir start/slutt-markør (default `<b>` og `</b>` for DH-lab-data).
   - Instruksjonen forutsetter at fragmentet er på formen `A<start>X<slutt>B` hvor `X` er målordet.

3. **Inputkilder**
   - **Lim inn**: én forekomst per linje. Teksten lagres også som `source_row["fragment"]`.
   - **Filopplasting**:
     - Plain `.txt`: tolkes som én forekomst per linje.
     - `.csv/.tsv`: `csv.Sniffer` + heuristikk for skilletegn. Brukeren velger fragmentkolonne (default `concordance` om den finnes).
     - Originalkolonner kopieres uendret til hver `source_row`.

4. **Sampling**
   - Brukeren kan kjøre *Kjør sample* (randomisert subset) eller *Kjør alt*.
   - Token-estimat vises for både full kjøring og valgt sample.

---

### 3. Promptkonstruksjon
- **Brukerprompt** (`user_prompt`): redigerbart felt som automatisk innholder beskrivelsen av target-markører og kategorioppsett.
- **Teknisk prompt** (`TECH_PROMPT`):
  - Angir formatkrav, JSON-struktur og feltregler.
  - For listefelt beskrives tydelig at de skal returneres som lister med maks 3 verdier.
  - Begge promptene kombineres til `prompt = user_prompt + "\n\n" + TECH_PROMPT`.

---

### 4. Kjøringsløp
1. **Recordbygging**
   - `build_records` lager rekkefølgebevarende `id`, `fragment`, `source_row`, `source_row_index`.
2. **Batching**
   - `chunks(records, BATCH_SIZE)` sender `"<id> | <fragment>"`-linjer til OpenAI.
3. **Parsing**
   - Modell-respons må være JSON med `items`.
   - `normalize_single_value`/`normalize_list_values` sikrer ren output per felt.
4. **Feilhåndtering**
   - Manglende `id` i respons → fallback-rad med `begrunnelse` = feiltekst og feltverdi = `"feil"` / `["feil"]`.
   - Unntak på batchnivå gir samme fallback for hele batchen.

---

### 5. Resultatpost-prosessering
1. **Rekkefølge**
   - Sorteres på `input_row_index`, deretter intern `id`.
   - Metadata-nøkler (`__luminoner_input_index`, `__luminoner_internal_id`) fjernes før visning/eksport.
2. **Statistikk**
   - For `unique`: teller én verdi per rad (tom → `(tom)`).
   - For `list`: hver verdi i listen teller individuelt (tom liste → `(tom liste)`).
3. **Eksport**
   - **JSONL**: én rad per annotert fragment, ekte lister for listefelt og `karakteristikker`.
   - **CSV**:
     - Kolonnerekkefølge: alle kategori-felt (i UI-rekkefølge) → `karakteristikker` → `begrunnelse` → originale kildekolonner (i samme rekkefølge som input) → eventuelle øvrige felt (modell, temperatur, etc.).
     - Listestrukturer serialiseres med `|`.

---

### 6. Migrasjon til React/JS
- **Frontend (React)**
  - Bygg komponenter for kategorioppsett, target-markører, input og kjøringskontroller basert på denne spesifikasjonen.
  - Gjenbruk JSON-datastrukturene for feltkonfigurasjon og resultater.
  - Hold brukerinstruks + teknisk prompt som ren tekst i UI, slik at de kan sendes til backend uendret.

- **Backend/API**
  - Implementer endepunkter for:
    - `POST /annotate` (mottar records + config, returnerer `all_rows` + evt. grafer/statistikk).
    - Valgfritt `POST /estimate` for token-estimat dersom man vil gjøre det server-side.
  - Rebruk logikken for batching, parsing og eksport (kan trekkes ut i modul).
  - Ikke lagre API-nøkler permanent; bruk request-scope.

- **Manifest/Architecture som sannhetskilde**
  - Beskriv nye felter/endringer først her og i `manifest.md`.
  - Frontend og backend kan generere sin egen dokumentasjon direkte fra disse spesifikasjonene.

---

### 7. Fremtidige utvidelser
- Gjøre maks-lengde for listefelt konfigurerbar.
- Tillate feltspesifikke instruksjoner og label-beskrivelser.
- Eget API-lag med kø og resultathistorikk.

---

*Sist oppdatert: 2025-12-05*

