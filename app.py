# app.py
import json, io, csv, time, random
from typing import List, Dict, Any
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Emitoner", layout="wide")

# ---------- Adgang (må ligge helt øverst) ----------
def gate():
    if st.session_state.get("authed"):
        return
    pw = st.text_input("Passord", type="password")
    if st.button("Logg inn"):
        if pw == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Feil passord.")
    st.stop()
gate()

st.title("Emitoner – batchannotering")

# ---------- Konfig ----------
API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)

colA, colB, colC, colD = st.columns([1.1,1,1,1.2])
with colA:
    MODEL = st.selectbox("Modell", ["gpt-4o-mini", "gpt-5-mini"], index=0)
with colB:
    BATCH_SIZE = st.number_input("Batch-størrelse", 10, 500, 100, 10)
with colC:
    TEMP = st.slider("temperature", 0.0, 1.0, 0.0, 0.1)
with colD:
    MAX_WORDS = st.number_input("Maks ord per linje", 5, 60, 25, 1)

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
    txt = st.text_area("Én forekomst per linje", height=220,
                       placeholder="fragment 1\nfragment 2\n...")
    if txt:
        lines = normalize_lines(txt.splitlines(), MAX_WORDS)
else:
    up = st.file_uploader("Last opp .txt/.csv/.tsv (én forekomst per linje eller kolonne)",
                          type=["txt","csv","tsv"])
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

st.caption(f"Fant {len(lines)} fragmenter (dupl/blanke kuttet). Maks {MAX_WORDS} ord per linje.")

# ---------- Instruks (system) ----------
st.subheader("2) Instruks (prompt)")
default_prompt = """
Du annoterer hvert fragment uavhengig (maks 25 ord brukt). Kategoriser som:
‘bokstavelig’ (vær/atmosfære/klimatiske forhold) 
‘metaforisk’ (debatt/stemning/klimaet i debatten). 
Gi også kort begrunnelse (≤15 ord) og confidence ∈ [0,1].
Returner KUN ÉN gyldig JSON med nøkkelen "items", der "items" er en liste av objekter i dette formatet:
{"id": '<int>',
 "kategori": "<kategori>",
 "karakteristikker": ["<andre relevante trekk"],
 "begrunnelse": "<kort forklaring (≤15 ord)>",
 "confidence": '<0–1>' 

 Behandle linjene uavhengig, behold id, og ikke legg til forklarende tekst eller markdown — kun selve JSON-objektet.

 """

prompt = st.text_area("System/oppgaveinstruks", value=default_prompt, height=160)

# ---------- Testmodus ----------
st.subheader("3) Estimat, test og kjøring")
col1, col2, col3 = st.columns([1,1,1.2])
with col1:
    testmode = st.checkbox("🧪 Testmodus (kjør liten batch først)")
with col2:
    n_test = st.number_input("Antall linjer i test", 5, 100, 10, 1, disabled=not testmode)
with col3:
    shuffle = st.checkbox("Tilfeldig utvalg i test", value=True, disabled=not testmode)

def choose_subset(all_lines: List[str]) -> List[str]:
    if not testmode or not all_lines:
        return all_lines
    idx = list(range(len(all_lines)))
    if shuffle:
        random.shuffle(idx)
    pick = sorted(idx[: int(n_test)])
    return [all_lines[i] for i in pick]

to_run = choose_subset(lines)

approx_in = len(to_run) * 40   # grovt anslag
approx_out = len(to_run) * 20
st.write(f"Grovt tokenestimat for **denne kjøringen**: "
         f"in ≈ {approx_in:,} · out ≈ {approx_out:,} · total ≈ {approx_in+approx_out:,}")

# ---------- Hjelpere ----------
def chunks(xs: List[Any], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

@st.cache_data(show_spinner=False)
def _ts():
    return time.strftime("%Y-%m-%dT%H-%M-%S")

def build_records(lines: List[str]) -> List[Dict[str, Any]]:
    # id = global rekkefølge i denne kjøringen
    return [{"id": i+1, "fragment": frag} for i, frag in enumerate(lines)]

def build_user_msg(batch: List[Dict[str, Any]]) -> str:
    # Modellvennlig, deterministisk struktur
    s = []
    for r in batch:
        s.append(f'{r["id"]} | {r["fragment"]}')
    return "\n".join(s)

def parse_items(raw_text: str) -> List[Dict[str, Any]]:
    data = json.loads(raw_text)
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
                temperature=TEMP,          # hold 0 for stabilitet
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

            # Map id -> fragment for berikelse
            frag_map = {r["id"]: r["fragment"] for r in batch}

            # Normaliser og berik
            for it in items:
                rid = it.get("id")
                row = {
                    "id": rid,
                    "fragment": frag_map.get(rid, ""),
                    "model": MODEL,
                    "temperature": TEMP,
                    # tillat 'kategori' eller 'bruk'
                    "kategori": it.get("kategori") or it.get("bruk") or "",
                    "karakteristikker": it.get("karakteristikker", []),
                    "begrunnelse": it.get("begrunnelse", ""),
                    "confidence": it.get("confidence", None),
                }
                all_rows.append(row)

            # Hvis modellen ikke returnerte alle id-ene, pad som feil
            got_ids = {it.get("id") for it in items}
            for r in batch:
                if r["id"] not in got_ids:
                    all_rows.append({
                        "id": r["id"], "fragment": r["fragment"],
                        "model": MODEL, "temperature": TEMP,
                        "kategori": "feil", "karakteristikker": [],
                        "begrunnelse": "manglende rad i svar", "confidence": 0.0
                    })

        except Exception as e:
            # Marker hele batchen som feil
            for r in batch:
                all_rows.append({
                    "id": r["id"], "fragment": r["fragment"],
                    "model": MODEL, "temperature": TEMP,
                    "kategori": "feil",
                    "karakteristikker": [],
                    "begrunnelse": str(e),
                    "confidence": 0.0
                })

        done += len(batch)
        progress.progress(done / total)
        status.write(f"Ferdig: {done}/{total}")

        # Testmodus: stopp etter første batch
        if testmode:
            break


    # Velg kolonnerekkefølge
    cols = ["id","fragment","kategori","karakteristikker","begrunnelse",
            "confidence","model","temperature"]
    fieldnames = [c for c in cols if c in all_rows[0]]  # faller tilbake på eksisterende
    
    # Oppsummering
    from collections import Counter
    counts = Counter([r.get("kategori","") for r in all_rows])
    st.table({"kategori": list(counts.keys()), "antall": list(counts.values())})
    
    # (valgfritt) enkel stolpe
    import pandas as pd
    dfc = pd.DataFrame({"kategori": list(counts.keys()), "antall": list(counts.values())})
    st.bar_chart(dfc.set_index("kategori"))

    
    ts = _ts()

    # Eksport (kun ved full kjøring eller hvis man ønsker i test også – her viser vi uansett)
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
        # flate ut lister i CSV
        o2 = {**o}
        if isinstance(o2.get("karakteristikker"), list):
            o2["karakteristikker"] = "|".join(o2["karakteristikker"])
        writer.writerow({k: o2.get(k, "") for k in fieldnames})
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    st.success("Kjøring ferdig ✅")
    if testmode:
        st.info("Testmodus: dette var kun første batch.")
    st.download_button("Last ned JSONL", data=jsonl_bytes,
                       file_name=f"emitoner_{ts}.jsonl", mime="application/jsonl")
    st.download_button("Last ned CSV", data=csv_bytes,
                       file_name=f"emitoner_{ts}.csv", mime="text/csv")
