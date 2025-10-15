# app.py
import json, io, csv, math, time
from typing import List
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Emitoner", layout="wide")
st.title("Emitoner – batchannotering")


def gate():
    if st.session_state.get("authed"): return
    pw = st.text_input("Passord", type="password")
    if st.button("Logg inn"):
        if pw == st.secrets["APP_PASSWORD"]:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Feil passord.")
    st.stop()
gate()

# --- Konfig ---
API_KEY = st.secrets["OPENAI_API_KEY"]  # legg nøkkelen i Secrets på share.streamlit
MODEL = st.selectbox("Modell", ["gpt-4o-mini", "gpt-5-mini"], index=0)
BATCH_SIZE = st.number_input("Batch-størrelse (linjer per kall)", 10, 500, 100, 10)
TEMP = st.slider("temperature", 0.0, 1.0, 0.0, 0.1)
MAX_WORDS = 25

client = OpenAI(api_key=API_KEY)




st.subheader("1) Data inn")
src = st.radio("Kilde", ["Lim inn", "Last opp CSV/TSV"], horizontal=True)

def normalize_lines(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        ln = (ln or "").strip()
        if not ln:
            continue
        # kutt til ~25 ord
        words = ln.split()
        if len(words) > MAX_WORDS:
            ln = " ".join(words[:MAX_WORDS])
        out.append(ln)
    # fjern duplikater
    return list(dict.fromkeys(out))

lines = []
if src == "Lim inn":
    txt = st.text_area("Én forekomst per linje", height=220, placeholder="fragment 1\nfragment 2\n...")
    if txt:
        lines = normalize_lines(txt.splitlines())
else:
    up = st.file_uploader("Last opp .txt/.csv/.tsv (én forekomst per linje eller kolonne)", type=["txt","csv","tsv"])
    if up:
        if up.type.startswith("text") or up.name.endswith(".txt"):
            lines = normalize_lines(up.read().decode("utf-8").splitlines())
        else:
            dialect = csv.excel if up.name.endswith(".csv") else csv.excel_tab
            reader = csv.reader(io.StringIO(up.read().decode("utf-8")), dialect=dialect)
            for row in reader:
                for cell in row:
                    lines.append(cell)
            lines = normalize_lines(lines)

st.caption(f"Fant {len(lines)} fragmenter (dupl/fyll kuttet). Maks {MAX_WORDS} ord per linje.")




st.subheader("2) Instruks (prompt)")
default_prompt = (
    "Oppgave: For hver linje, klassifiser som ‘bokstavelig’ (vær/atmosfære) "
    "eller ‘metaforisk’ (debatt/stemning osv.).\n"
    "Format: Svar KUN som JSONL (én JSON per linje) med nøkler: "
    '{"bruk":"bokstavelig|metaforisk","begrunnelse":"<=15 ord","confidence":0.0-1.0}\n'
    "Viktig: Behandle linjene uavhengig, bruk KUN teksten i linjen."
)
prompt = st.text_area("System/oppgaveinstruks", value=default_prompt, height=160)

st.subheader("3) Estimat & kjøring")
approx_in = len(lines) * 40  # grovt anslag
approx_out = len(lines) * 20
st.write(f"Grovt tokenestimat: in ≈ {approx_in:,} · out ≈ {approx_out:,} · total ≈ {approx_in+approx_out:,}")

def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

run = st.button("Kjør annotering")

@st.cache_data(show_spinner=False)
def _ts():
    return time.strftime("%Y-%m-%dT%H-%M-%S")

if run and lines:
    st.info("Starter kjøring…")
    all_json = []
    progress = st.progress(0.0)
    status = st.empty()
    done = 0

    for batch in chunks(lines, BATCH_SIZE):
        # Pakk batch til en enkelt chat-forespørsel
        # Vi ber modellen returnere JSONL, én linje per input.
        user_msg = "Annotér følgende linjer som beskrevet. Returner Nøyaktig N JSON-objekter i JSONL, i samme rekkefølge.\n"
        for i, ln in enumerate(batch, 1):
            user_msg += f"{i}. {ln}\n"

        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=TEMP,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg},
                ]
            )
            text = r.choices[0].message.content.strip()
            # Parse JSONL
            parsed = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # fiks ev. trailing komma/markdown
                line = line.strip("`").strip()
                obj = json.loads(line)
                parsed.append(obj)

            # Sikre samme lengde
            if len(parsed) != len(batch):
                # fallback: pad med uklar
                for _ in range(len(batch) - len(parsed)):
                    parsed.append({"bruk":"uklar","begrunnelse":"","confidence":0.0})
            # berik med input
            for frag, lab in zip(batch, parsed):
                lab["fragment"] = frag
                lab["model"] = MODEL
                lab["temperature"] = TEMP
                all_json.append(lab)

        except Exception as e:
            # marker feil for denne batchen
            for frag in batch:
                all_json.append({
                    "fragment": frag,
                    "model": MODEL,
                    "temperature": TEMP,
                    "bruk": "feil",
                    "begrunnelse": str(e),
                    "confidence": 0.0
                })

        done += len(batch)
        progress.progress(done / len(lines))
        status.write(f"Ferdig: {done}/{len(lines)}")

    ts = _ts()
    # JSONL
    jsonl_buf = io.StringIO()
    for obj in all_json:
        jsonl_buf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    jsonl_bytes = jsonl_buf.getvalue().encode("utf-8")

    # CSV
    csv_buf = io.StringIO()
    fieldnames = sorted({k for o in all_json for k in o.keys()})
    writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
    writer.writeheader()
    for o in all_json:
        writer.writerow({k: o.get(k, "") for k in fieldnames})
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    st.success("Kjøring ferdig ✅")
    st.download_button("Last ned JSONL", data=jsonl_bytes, file_name=f"emitoner_{ts}.jsonl", mime="application/jsonl")
    st.download_button("Last ned CSV", data=csv_bytes, file_name=f"emitoner_{ts}.csv", mime="text/csv")
