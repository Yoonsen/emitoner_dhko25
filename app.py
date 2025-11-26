# app.py
import json, io, csv, time, random
from typing import List, Dict, Any
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Luminoner / emitoner", layout="wide")

# ---------- Adgang ----------
import os

def gate():
    if st.session_state.get("authed"):
        return

    APP_PASSWORD = (
        st.secrets.get("APP_PASSWORD")
        if "APP_PASSWORD" in st.secrets
        else os.getenv("APP_PASSWORD", "lokaltest")   # fallback lokalt
    )

    pw = st.text_input("Passord", type="password")
    if st.button("Logg inn"):
        if pw == APP_PASSWORD:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Feil passord.")
    st.stop()

gate()

st.title("Luminoner – batchannotering")

# ---------- Konfig ----------
API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
with colA:
    MODEL = st.selectbox(
        "Modell",
        ["gpt-4o-mini", "gpt-5-mini", "gpt-4"],
        index=0,
        help="gpt-4 er dyrere – bruk den kun på små tester."
    )
with colB:
    BATCH_SIZE = st.number_input("Batch-størrelse", 10, 500, 10, 10)
with colC:
    TEMP = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
with colD:
    MAX_WORDS = st.number_input("Maks ord per linje", 5, 60, 25, 1)

categories_str = st.text_input(
    "Kategorier (kommaseparert for feltet «kategori»)",
    "bokstavelig, metaforisk",
    help="Du kan f.eks. bruke «personlig, upersonlig, uklar» for personreferanser."
)
CATEGORIES = [c.strip() for c in categories_str.split(",") if c.strip()]
if CATEGORIES:
    categories_display = ", ".join(f'"{c}"' for c in CATEGORIES)
else:
    categories_display = '"kategori1", "kategori2"'

# ---------- Data inn ----------
st.subheader("1) Data inn")
src = st.radio("Kilde", ["Lim inn", "Last opp CSV/TSV"], horizontal=True)


def normalize_lines(lines: List[str], max_words: int) -> List[str]:
    out = []
    for ln in lines:
        ln = (ln or "").strip()
        if not ln:
            continue
        words = ln.split()
        if len(words) > max_words:
            ln = " ".join(words[:max_words])
        out.append(ln)
    # fjern duplikater, behold rekkefølge
    return list(dict.fromkeys(out))


lines: List[str] = []
if src == "Lim inn":
    txt = st.text_area(
        "Én forekomst per linje (rå konkordanser)",
        height=220,
        placeholder="fragment 1\nfragment 2\n."
    )
    if txt:
        lines = normalize_lines(txt.splitlines(), MAX_WORDS)
else:
    up = st.file_uploader(
        "Last opp .txt/.csv/.tsv (én forekomst per linje eller kolonne)",
        type=["txt", "csv", "tsv"],
    )
    if up:
        if up.type.startswith("text") or up.name.endswith(".txt"):
            lines = normalize_lines(up.read().decode("utf-8").splitlines(), MAX_WORDS)
        else:
            dialect = csv.excel if up.name.endswith(".csv") else csv.excel_tab
            reader = csv.reader(io.StringIO(up.read().decode("utf-8")), dialect=dialect)
            cells = []
            for row in reader:
                for cell in row:
                    cells.append(cell)
            lines = normalize_lines(cells, MAX_WORDS)

st.caption(
    f"Fant {len(lines)} fragmenter (dupl/blanke kuttet). Maks {MAX_WORDS} ord per linje."
)

# ---------- Instruks (system) ----------
st.subheader("2) Instruks (oppgavebeskrivelse)")

default_user_prompt = f"""
Du annoterer hvert tekstfragment uavhengig.

Bruk feltet "kategori" til å tildele én av følgende koder:
{categories_display}

Bruk feltet "karakteristikker" til 0–3 korte stikkord som sier noe om fenomenet
du undersøker (f.eks. «personlig», «offentlig», «historisk», «ironisk», osv.).

Du kan bruke denne appen til f.eks.:
- forskjell på fysisk klima vs. debattklima
- typer personreferanser
- andre typer luminoner med faste koder i "kategori".
""".strip()

user_prompt = st.text_area(
    "Oppgavebeskrivelse (kan endres for andre luminoner)",
    value=default_user_prompt,
    height=200,
)

TECH_PROMPT = f"""
Formatkrav (viktig):

- Du får linjer på formen "<id> | <fragment>".
- Du skal behandle hvert fragment uavhengig.
- Feltet "kategori" skal være én av: {categories_display}.
- Du skal alltid svare med KUN ÉN gyldig JSON-struktur med nøkkelen "items".
- "items" skal være en liste med objekter på denne formen:

  {{
    "id": <int>,                         // samme id som i input
    "kategori": <én av {categories_display}>,
    "karakteristikker": ["...", "..."],  // 0–3 korte stikkord
    "begrunnelse": "<maks 15 ord>",
    "confidence": <tall mellom 0 og 1>   // ikke streng, men et tall
  }}

- Ikke legg til annen tekst, forklaringer eller markdown utenfor dette ene JSON-objektet.
- Behold alle id-er du får, og ikke oppfinn nye.
"""

with st.expander("Tekniske formatkrav (JSON)", expanded=False):
    st.code(TECH_PROMPT)

# dette er faktiske systemprompt
prompt = user_prompt.strip() + "\n\n" + TECH_PROMPT

# ---------- Testmodus ----------
st.subheader("3) Estimat, test og kjøring")
col1, col2, col3 = st.columns([1, 1, 1.2])
with col1:
    testmode = st.checkbox("🧪 Testmodus (kjør liten batch først)")
with col2:
    n_test = st.number_input(
        "Antall linjer i test", 5, 100, 10, 1, disabled=not testmode
    )
with col3:
    shuffle = st.checkbox(
        "Tilfeldig utvalg i test", value=True, disabled=not testmode
    )


def choose_subset(all_lines: List[str]) -> List[str]:
    if not testmode or not all_lines:
        return all_lines
    idx = list(range(len(all_lines)))
    if shuffle:
        random.shuffle(idx)
    pick = sorted(idx[: int(n_test)])
    return [all_lines[i] for i in pick]


to_run = choose_subset(lines)

approx_in = len(to_run) * 40  # grovt anslag
approx_out = len(to_run) * 20
st.write(
    f"Grovt tokenestimat for **denne kjøringen**: "
    f"in ≈ {approx_in:,} · out ≈ {approx_out:,} · total ≈ {approx_in+approx_out:,}"
)

# ---------- Hjelpere ----------
def chunks(xs: List[Any], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


@st.cache_data(show_spinner=False)
def _ts():
    return time.strftime("%Y-%m-%dT%H-%M-%S")


def build_records(lines: List[str]) -> List[Dict[str, Any]]:
    # id = global rekkefølge i denne kjøringen
    return [{"id": i + 1, "fragment": frag} for i, frag in enumerate(lines)]


def build_user_msg(batch: List[Dict[str, Any]]) -> str:
    # Modellvennlig, deterministisk struktur
    s = []
    for r in batch:
        s.append(f'{r["id"]} | {r["fragment"]}')
    return "\n".join(s)


def parse_items(raw_text: str) -> List[Dict[str, Any]]:
    """
    Litt mer robust:
    - prøv ren JSON først
    - hvis det feiler, trekk ut første { ... siste } og prøv igjen
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Kunne ikke finne gyldig JSON-objekt i svaret.")
        data = json.loads(raw_text[start : end + 1])

    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("JSON mangler 'items' som liste.")
    return items


# ---------- Kjøring ----------
run = st.button("Kjør annotering")
if run and to_run:
    st.info("Starter kjøring…")
    all_rows: List[Dict[str, Any]] = []
    recs = build_records(to_run)
    total = len(recs)
    progress = st.progress(0.0)
    status = st.empty()
    done = 0
    batch_counter = 0

    for batch in chunks(recs, int(BATCH_SIZE)):
        batch_counter += 1
        user_msg = build_user_msg(batch)

        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=TEMP,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = r.choices[0].message.content.strip()

            try:
                items = parse_items(text)
            except Exception as e_json:
                with st.expander(f"JSON-feil (batch {batch_counter}) – rå svar"):
                    st.code(text)
                raise e_json

            frag_map = {r["id"]: r["fragment"] for r in batch}

            for it in items:
                rid = it.get("id")
                row = {
                    "id": rid,
                    "fragment": frag_map.get(rid, ""),
                    "model": MODEL,
                    "temperature": TEMP,
                    "kategori": it.get("kategori") or it.get("bruk") or "",
                    "karakteristikker": it.get("karakteristikker", []),
                    "begrunnelse": it.get("begrunnelse", ""),
                    "confidence": it.get("confidence", None),
                }
                all_rows.append(row)

            got_ids = {it.get("id") for it in items}
            for r in batch:
                if r["id"] not in got_ids:
                    all_rows.append(
                        {
                            "id": r["id"],
                            "fragment": r["fragment"],
                            "model": MODEL,
                            "temperature": TEMP,
                            "kategori": "feil",
                            "karakteristikker": [],
                            "begrunnelse": "manglende rad i svar",
                            "confidence": 0.0,
                        }
                    )

        except Exception as e:
            for r in batch:
                all_rows.append(
                    {
                        "id": r["id"],
                        "fragment": r["fragment"],
                        "model": MODEL,
                        "temperature": TEMP,
                        "kategori": "feil",
                        "karakteristikker": [],
                        "begrunnelse": str(e),
                        "confidence": 0.0,
                    }
                )

        done += len(batch)
        progress.progress(done / total)
        status.write(f"Ferdig: {done}/{total}")

        if testmode:
            break

    if not all_rows:
        st.warning("Ingen rader å vise.")
    else:
        from collections import Counter
        import pandas as pd

        counts = Counter([r.get("kategori", "") for r in all_rows])
        st.table({"kategori": list(counts.keys()), "antall": list(counts.values())})

        dfc = pd.DataFrame({"kategori": list(counts.keys()), "antall": list(counts.values())})
        st.bar_chart(dfc.set_index("kategori"))

        ts = _ts()

        # JSONL
        jsonl_buf = io.StringIO()
        for obj in all_rows:
            jsonl_buf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        jsonl_bytes = jsonl_buf.getvalue().encode("utf-8")

        # CSV
        csv_buf = io.StringIO()
        fieldnames = sorted({k for o in all_rows for k in o.keys()})
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for o in all_rows:
            o2 = {**o}
            if isinstance(o2.get("karakteristikker"), list):
                o2["karakteristikker"] = "|".join(o2["karakteristikker"])
            writer.writerow({k: o2.get(k, "") for k in fieldnames})
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        st.success("Kjøring ferdig ✅")
        if testmode:
            st.info("Testmodus: dette var kun første batch.")
        st.download_button(
            "Last ned JSONL",
            data=jsonl_bytes,
            file_name=f"luminoner_{ts}.jsonl",
            mime="application/jsonl",
        )
        st.download_button(
            "Last ned CSV",
            data=csv_bytes,
            file_name=f"luminoner_{ts}.csv",
            mime="text/csv",
        )
