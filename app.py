# app.py
import json, io, csv, time, random
from typing import List, Dict, Any
import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="Luminoner / emitoner", layout="wide")


def secret_or_env(key: str, default: Any = None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


CATCH_ALL_VALUE = "uten-relevans"
INITIAL_CATEGORY_FIELDS = [
    {
        "id": 0,
        "label": "kategori",
        "values": "bokstavelig, metaforisk",
        "mode": "unique",
    },
]

META_ROW_INDEX_KEY = "__luminoner_input_index"
META_RECORD_ID_KEY = "__luminoner_internal_id"
CATEGORY_MODE_LABELS = {"unique": "Unik", "list": "Liste"}
DEFAULT_TARGET_MARKER_LEFT = "<b>"
DEFAULT_TARGET_MARKER_RIGHT = "</b>"

if "last_source_headers" not in st.session_state:
    st.session_state["last_source_headers"] = []


st.title("Luminoner – batchannotering")

# ---------- Konfig ----------
API_KEY = secret_or_env("OPENAI_API_KEY")
if not API_KEY:
    st.error("Manglende OPENAI_API_KEY i .streamlit/secrets.toml eller miljøvariabel.")
    st.stop()
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

st.subheader("Kategorioppsett (kolonner)")
st.caption(
    "Del opp annoteringen i flere felter (f.eks. «sport», «økonomi», «konflikt»). "
    "Hvert felt får et eget sett med lovlige verdier."
)
st.caption(
    f"Verdien «{CATCH_ALL_VALUE}» legges automatisk til alle felter som en catch-all for "
    "fragmenter uten treff."
)

if "category_field_entries" not in st.session_state:
    st.session_state["category_field_entries"] = [
        dict(entry) for entry in INITIAL_CATEGORY_FIELDS
    ]
    st.session_state["category_field_counter"] = len(INITIAL_CATEGORY_FIELDS)

entries = st.session_state["category_field_entries"]

if "target_marker_left_input" not in st.session_state:
    st.session_state["target_marker_left_input"] = DEFAULT_TARGET_MARKER_LEFT
if "target_marker_right_input" not in st.session_state:
    st.session_state["target_marker_right_input"] = DEFAULT_TARGET_MARKER_RIGHT

action_cols = st.columns([0.25, 0.75])
with action_cols[0]:
    if st.button("➕ Legg til felt", use_container_width=True):
        next_id = st.session_state.get("category_field_counter", len(entries))
        entries.append({"id": next_id, "label": "", "values": "", "mode": "unique"})
        st.session_state["category_field_counter"] = next_id + 1
with action_cols[1]:
    st.caption("Bruk «Fjern» for å ta bort et felt (minst ett felt må eksistere).")

category_fields: List[Dict[str, Any]] = []
used_keys = set()
for idx, entry in enumerate(entries):
    col_label, col_values, col_mode, col_remove = st.columns([1, 2, 0.8, 0.25])
    label_val = col_label.text_input(
        "Feltnavn",
        value=entry.get("label", ""),
        placeholder="f.eks. Sport",
        key=f"category_field_label_{entry['id']}",
    ).strip()
    values_val = col_values.text_area(
        "Tillatte verdier (kommaseparert)",
        value=entry.get("values", ""),
        placeholder="fotball, svømming",
        help="Separér alternativene med komma eller linjeskift.",
        height=80,
        key=f"category_field_values_{entry['id']}",
    )
    mode_default = entry.get("mode", "unique")
    mode_val = col_mode.radio(
        "Variant",
        options=["unique", "list"],
        index=0 if mode_default != "list" else 1,
        format_func=lambda opt: CATEGORY_MODE_LABELS.get(opt, opt),
        key=f"category_field_mode_{entry['id']}",
        horizontal=True,
        help="Unik = én verdi. Liste = 0–3 verdier fra samme vokabular.",
    )
    entry["mode"] = mode_val
    if col_remove.button(
        "Fjern",
        key=f"remove_category_field_{entry['id']}",
        use_container_width=True,
        disabled=len(entries) == 1,
    ):
        del entries[idx]
        st.rerun()

    display_label = label_val or f"Felt {idx + 1}"
    field_key = display_label
    base_key = field_key
    suffix = 2
    while field_key in used_keys:
        field_key = f"{base_key}_{suffix}"
        suffix += 1
    used_keys.add(field_key)

    tokens: List[str] = []
    for line in values_val.splitlines():
        tokens.extend(t.strip() for t in line.split(","))
    field_values = [t for t in tokens if t]
    if CATCH_ALL_VALUE not in field_values:
        field_values.append(CATCH_ALL_VALUE)

    category_fields.append(
        {
            "label": display_label,
            "key": field_key,
            "values": field_values,
            "mode": mode_val,
        }
    )


def _values_display(values: List[str]) -> str:
    return ", ".join(f'"{v}"' for v in values) if values else '"verdi1", "verdi2"'


def _list_example(values: List[str]) -> str:
    if not values:
        return '["verdi1", "verdi2"]'
    sample = values[: min(2, len(values))]
    inner = ", ".join(f'"{v}"' for v in sample)
    return f"[{inner}]"


field_names_display = ", ".join(f'"{c["label"]}"' for c in category_fields)
if not field_names_display:
    field_names_display = '"kategori"'
field_rules_lines: List[str] = []
for c in category_fields:
    if c["mode"] == "list":
        field_rules_lines.append(
            f'- Feltet "{c["key"]}" kan inneholde opptil 3 verdier valgt fra: '
            f'{_values_display(c["values"])} (returner som liste).'
        )
    else:
        field_rules_lines.append(
            f'- Feltet "{c["key"]}" skal være én av: {_values_display(c["values"])}.'
        )
if not field_rules_lines:
    field_rules_lines = [
        '- Feltet "kategori" skal være én av: "kategori1", "kategori2".'
    ]
field_rules_lines.append(
    f'- Hvis ingen kode passer i et felt, bruk verdien "{CATCH_ALL_VALUE}".'
)
field_rules_text = "\n".join(field_rules_lines)
json_field_line_parts: List[str] = []
for c in category_fields:
    if c["mode"] == "list":
        json_field_line_parts.append(
            f'    "{c["key"]}": {_list_example(c["values"])},  '
            "// liste med 0–3 verdier fra settet over"
        )
    else:
        json_field_line_parts.append(
            f'    "{c["key"]}": <én av {_values_display(c["values"])}>,'
        )
json_field_lines = "\n".join(json_field_line_parts)
if not json_field_lines:
    json_field_lines = '    "kategori": <én av "kategori1", "kategori2">,'  # fallback

# ---------- Data inn ----------
st.subheader("1) Data inn")
src = st.radio("Kilde", ["Lim inn", "Last opp CSV/TSV"], horizontal=True)


def clamp_fragment(text: str, max_words: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def normalize_lines(lines: List[str], max_words: int) -> List[str]:
    out = []
    for ln in lines:
        cleaned = clamp_fragment(ln, max_words)
        if not cleaned:
            continue
        out.append(cleaned)
    # fjern duplikater, behold rekkefølge
    return list(dict.fromkeys(out))


def detect_delimiter(raw_text: str, fallback: str = ",") -> str:
    """
    Best-effort deteksjon av tabellskilletegn (komma, semikolon, tab, pipe).
    Bruker csv.Sniffer når mulig og faller ellers tilbake til enkel telling.
    """
    sample = raw_text[:5000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        if dialect.delimiter:
            return dialect.delimiter
    except csv.Error:
        pass

    counts = {d: raw_text.count(d) for d in [",", ";", "\t", "|"]}
    best = max(counts, key=lambda d: counts[d])
    if counts.get(best, 0) > 0:
        return best
    return fallback


def pick_sample(
    entries: List[Dict[str, Any]], sample_size: int, shuffle: bool = True
) -> List[Dict[str, Any]]:
    """
    Velg et delsett på sample_size rader. Rader beholdes i original rekkefølge.
    """
    total = len(entries)
    if sample_size <= 0:
        return []
    if sample_size >= total:
        return entries
    idx = list(range(total))
    if shuffle:
        random.shuffle(idx)
    chosen = sorted(idx[:sample_size])
    return [entries[i] for i in chosen]


def normalize_list_values(value: Any, max_items: int = 3) -> List[str]:
    """
    Sørger for at listefelt alltid blir en liste med rene strenger (maks max_items).
    """
    if isinstance(value, list):
        items = value
    elif value is None:
        items = []
    else:
        items = [value]
    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def normalize_single_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if value is None:
        return ""
    return str(value).strip()


input_entries: List[Dict[str, Any]] = []
selected_fragment_column: str | None = None
current_source_headers: List[str] = []

if src == "Lim inn":
    txt = st.text_area(
        "Én forekomst per linje (rå konkordanser)",
        height=220,
        placeholder="fragment 1\nfragment 2\n.",
    )
    if txt:
        normalized = normalize_lines(txt.splitlines(), MAX_WORDS)
        if normalized:
            current_source_headers = ["fragment"]
        for idx, frag in enumerate(normalized):
            input_entries.append(
                {
                    "fragment": frag,
                    "source_row": {"fragment": frag},
                    "source_row_index": idx + 1,
                }
            )
else:
    up = st.file_uploader(
        "Last opp .txt/.csv/.tsv (én forekomst per linje eller kolonne)",
        type=["txt", "csv", "tsv"],
    )
    if up:
        file_bytes = up.getvalue()
        name_lower = up.name.lower()
        is_tsv = name_lower.endswith(".tsv") or name_lower.endswith(".tab")
        is_csv = name_lower.endswith(".csv")
        is_table_file = is_csv or is_tsv

        if not is_table_file:
            normalized = normalize_lines(
                file_bytes.decode("utf-8", errors="ignore").splitlines(),
                MAX_WORDS,
            )
            if normalized:
                current_source_headers = ["fragment"]
            for idx, frag in enumerate(normalized):
                input_entries.append(
                    {
                        "fragment": frag,
                        "source_row": {"fragment": frag},
                        "source_row_index": idx + 1,
                    }
                )
        else:
            csv_text = file_bytes.decode("utf-8", errors="ignore")
            delimiter = "\t" if is_tsv else detect_delimiter(csv_text, ",")
            reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
            rows = list(reader)
            headers = reader.fieldnames or []

            if not headers:
                st.error("Fant ingen kolonner i filen. Sjekk at CSV/TSV har header-rad.")
            else:
                current_source_headers = headers
                default_idx = 0
                for i, header in enumerate(headers):
                    if (header or "").strip().lower() == "concordance":
                        default_idx = i
                        break

                selected_fragment_column = st.selectbox(
                    "Kolonne med fragmenter",
                    options=headers,
                    index=default_idx,
                    key="fragment_column_select",
                    help='Standard er "concordance" hvis den finnes.',
                )

                for idx, row in enumerate(rows):
                    row_copy = {h: row.get(h, "") for h in headers}
                    frag_value = row_copy.get(selected_fragment_column, "")
                    cleaned = clamp_fragment(frag_value, MAX_WORDS)
                    input_entries.append(
                        {
                            "fragment": cleaned,
                            "source_row": row_copy,
                            "source_row_index": idx + 1,
                        }
                    )

                empty_count = sum(1 for entry in input_entries if not entry["fragment"])
                if empty_count:
                    st.warning(
                        f"{empty_count} rad(er) mangler tekst i kolonnen «{selected_fragment_column}»."
                    )

st.session_state["last_source_headers"] = current_source_headers

if selected_fragment_column:
    st.caption(
        f"Fant {len(input_entries)} rader (kolonne «{selected_fragment_column}»). "
        f"Originale kolonner beholdes, og fragmenter kuttes til maks {MAX_WORDS} ord for modellkallet."
    )
else:
    st.caption(
        f"Fant {len(input_entries)} fragmenter (dupl/blanke kuttet). Maks {MAX_WORDS} ord per linje."
    )

st.subheader("1b) Target-markering (målord)")
st.caption(
    "Marker målordet/uttrykket du vil klassifisere med en start- og sluttmarkør. "
    "Standard matcher DH-lab-formatet med <b>mål</b>."
)
marker_cols = st.columns(2)
target_marker_left_raw = marker_cols[0].text_input(
    "Startmarkør",
    key="target_marker_left_input",
    help="Tegnene rett før målfragmentet (f.eks. <b>).",
    max_chars=20,
)
target_marker_right_raw = marker_cols[1].text_input(
    "Sluttmarkør",
    key="target_marker_right_input",
    help="Tegnene rett etter målfragmentet (f.eks. </b>).",
    max_chars=20,
)
target_marker_left = (target_marker_left_raw or DEFAULT_TARGET_MARKER_LEFT).strip()
target_marker_right = (target_marker_right_raw or DEFAULT_TARGET_MARKER_RIGHT).strip()
target_marker_example = f"A{target_marker_left}klima{target_marker_right}B"
st.caption("Eksempel på forventet struktur (A = venstre kontekst, B = høyre kontekst):")
st.code(target_marker_example)

# ---------- Instruks (system) ----------
st.subheader("2) Instruks (oppgavebeskrivelse)")

default_user_prompt = f"""
Du annoterer hvert tekstfragment uavhengig.

Fragmentene har formen A{target_marker_left}X{target_marker_right}B der X (mellom
markørene) er målordet du skal beskrive. Bruk konteksten før/etter som støtte,
men alle kategorier skal gjelde selve X.

Bruk kategorifeltene {field_names_display} til å fordele én kode per felt (f.eks.
sport/økonomi/konflikt).

Hvis ingen kode passer i et felt, bruk verdien "{CATCH_ALL_VALUE}".

Bruk feltet "karakteristikker" til 0–3 korte stikkord som sier noe om fenomenet
du undersøker (f.eks. «personlig», «offentlig», «historisk», «ironisk», osv.).

Du kan bruke denne appen til f.eks.:
- forskjell på fysisk klima vs. debattklima
- typer personreferanser
- andre typer luminoner med faste koder per felt.
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
- Fragmentene følger mønsteret A{target_marker_left}X{target_marker_right}B – X
  (mellom markørene) er målfragmentet du klassifiserer.
- Bruk konteksten utenfor markørene som støtte, men feltverdiene skal beskrive X.
{field_rules_text}
- Du skal alltid svare med KUN ÉN gyldig JSON-struktur med nøkkelen "items".
- "items" skal være en liste med objekter på denne formen:

  {{
    "id": <int>,                         // samme id som i input
{json_field_lines}
    "karakteristikker": ["...", "..."],  // 0–3 korte stikkord
    "begrunnelse": "<maks 15 ord>"
  }}

- Ikke legg til annen tekst, forklaringer eller markdown utenfor dette ene JSON-objektet.
- Behold alle id-er du får, og ikke oppfinn nye.
"""

with st.expander("Tekniske formatkrav (JSON)", expanded=False):
    st.code(TECH_PROMPT)

# dette er faktiske systemprompt
prompt = user_prompt.strip() + "\n\n" + TECH_PROMPT

# ---------- Sample og kjøring ----------
st.subheader("3) Estimat og kjøring")
entries_count = len(input_entries)
sample_disabled = entries_count == 0
sample_max_value = entries_count or 1
sample_default = min(10, sample_max_value)
sample_cols = st.columns([1, 1])
with sample_cols[0]:
    sample_size = st.number_input(
        "Antall linjer i sample",
        min_value=1,
        max_value=sample_max_value,
        value=sample_default,
        step=1,
        disabled=sample_disabled,
        help="Velg hvor mange rader som brukes når du kjører sample.",
    )
with sample_cols[1]:
    sample_shuffle = st.checkbox(
        "Tilfeldig sample",
        value=True,
        disabled=sample_disabled,
        help="Når aktivert velges samplet tilfeldig før det sorteres.",
    )

run_cols = st.columns([1, 1])
with run_cols[0]:
    run_all = st.button(
        "Kjør alt", type="primary", use_container_width=True, disabled=sample_disabled
    )
with run_cols[1]:
    run_sample = st.button(
        "Kjør sample", use_container_width=True, disabled=sample_disabled
    )

entries_to_process: List[Dict[str, Any]] | None = None
run_mode = None
if run_all and entries_count:
    entries_to_process = input_entries
    run_mode = "all"
elif run_sample and entries_count:
    sample_target = min(int(sample_size), entries_count)
    entries_to_process = pick_sample(input_entries, sample_target, sample_shuffle)
    run_mode = "sample"

if entries_count:
    approx_all_in = entries_count * 40
    approx_all_out = entries_count * 20
    st.write(
        f"Grovt tokenestimat for **alle data**: "
        f"in ≈ {approx_all_in:,} · out ≈ {approx_all_out:,} · "
        f"total ≈ {approx_all_in + approx_all_out:,}"
    )
    sample_preview = min(int(sample_size), entries_count)
    approx_sample_in = sample_preview * 40
    approx_sample_out = sample_preview * 20
    st.caption(
        f"Et sample på {sample_preview} rader vil bruke ca. "
        f"in ≈ {approx_sample_in:,} · out ≈ {approx_sample_out:,} tokens."
    )
else:
    st.caption("Ingen data ennå – last opp eller lim inn fragmenter for å starte.")

# ---------- Hjelpere ----------
def chunks(xs: List[Any], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


@st.cache_data(show_spinner=False)
def _ts():
    return time.strftime("%Y-%m-%dT%H-%M-%S")


def build_records(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # id = lokal rekkefølge i denne kjøringen (beholder original rekkefølge via source_row_index)
    records: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        records.append(
            {
                "id": idx + 1,
                "fragment": entry.get("fragment", ""),
                "source_row": entry.get("source_row"),
                "source_row_index": entry.get("source_row_index", idx + 1),
            }
        )
    return records


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
to_run_entries = entries_to_process or []
if to_run_entries:
    run_desc = "sample" if run_mode == "sample" else "alle rader"
    st.info(f"Starter kjøring ({run_desc})…")
    all_rows: List[Dict[str, Any]] = []
    recs = build_records(to_run_entries)
    total = len(recs)
    progress = st.progress(0.0)
    status = st.empty()
    done = 0
    batch_counter = 0
    record_lookup = {rec["id"]: rec for rec in recs}

    def compose_row(
        record_id: int, fragment_value: str, record: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        rec = record or record_lookup.get(record_id, {"id": record_id})
        source_row = rec.get("source_row") or {}
        base = dict(source_row)
        idx = rec.get("source_row_index")
        if idx is not None:
            base[META_ROW_INDEX_KEY] = idx
        base[META_RECORD_ID_KEY] = rec.get("id", record_id)
        base["model"] = MODEL
        base["temperature"] = TEMP
        return base

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
                if rid is None:
                    continue
                row = compose_row(rid, frag_map.get(rid, ""))
                row["karakteristikker"] = normalize_list_values(
                    it.get("karakteristikker", [])
                )
                row["begrunnelse"] = normalize_single_value(it.get("begrunnelse", ""))
                for field in category_fields:
                    if field["mode"] == "list":
                        row[field["key"]] = normalize_list_values(
                            it.get(field["key"], [])
                        )
                    else:
                        row[field["key"]] = normalize_single_value(
                            it.get(field["key"], "")
                        )
                all_rows.append(row)

            got_ids = {it.get("id") for it in items}
            for r in batch:
                if r["id"] not in got_ids:
                    row = compose_row(r["id"], r["fragment"], record=r)
                    row["karakteristikker"] = []
                    row["begrunnelse"] = "manglende rad i svar"
                    for field in category_fields:
                        if field["mode"] == "list":
                            row[field["key"]] = ["feil"]
                        else:
                            row[field["key"]] = "feil"
                    all_rows.append(row)

        except Exception as e:
            for r in batch:
                row = compose_row(r["id"], r["fragment"], record=r)
                row["karakteristikker"] = []
                row["begrunnelse"] = str(e)
                for field in category_fields:
                    if field["mode"] == "list":
                        row[field["key"]] = ["feil"]
                    else:
                        row[field["key"]] = "feil"
                all_rows.append(row)

        done += len(batch)
        progress.progress(done / total)
        status.write(f"Ferdig: {done}/{total}")

    if all_rows:
        def _row_sort_key(row: Dict[str, Any]):
            idx = row.get(META_ROW_INDEX_KEY)
            if idx is None:
                idx = row.get(META_RECORD_ID_KEY, 0)
            return (idx, row.get(META_RECORD_ID_KEY, 0))

        all_rows.sort(key=_row_sort_key)
        for row in all_rows:
            row.pop(META_ROW_INDEX_KEY, None)
            row.pop(META_RECORD_ID_KEY, None)

    if not all_rows:
        st.warning("Ingen rader å vise.")
    else:
        from collections import Counter
        import pandas as pd

        if not category_fields:
            st.warning("Ingen kategorifelter definert – oppdater oppsettet over.")
        else:
            for cat in category_fields:
                values_for_counts: List[str] = []
                empty_label = "(tom)"
                if cat["mode"] == "list":
                    empty_label = "(tom liste)"
                    for r in all_rows:
                        cell = r.get(cat["key"])
                        if isinstance(cell, list) and cell:
                            for val in cell:
                                text = str(val).strip()
                                values_for_counts.append(text or empty_label)
                        else:
                            values_for_counts.append(empty_label)
                else:
                    for r in all_rows:
                        text = normalize_single_value(r.get(cat["key"], ""))
                        values_for_counts.append(text or empty_label)

                counts = Counter(values_for_counts)
                st.markdown(f"**Fordeling for {cat['label']}**")
                st.table({"verdi": list(counts.keys()), "antall": list(counts.values())})
                if counts:
                    dfc = pd.DataFrame(
                        {"verdi": list(counts.keys()), "antall": list(counts.values())}
                    )
                    st.bar_chart(dfc.set_index("verdi"))

        ts = _ts()

        # JSONL
        jsonl_buf = io.StringIO()
        for obj in all_rows:
            jsonl_buf.write(json.dumps(obj, ensure_ascii=False) + "\n")
        jsonl_bytes = jsonl_buf.getvalue().encode("utf-8")

        # CSV
        csv_buf = io.StringIO()
        ordered_keys: List[str] = []
        for o in all_rows:
            for k in o.keys():
                if k not in ordered_keys:
                    ordered_keys.append(k)
        analysis_order = [c["key"] for c in category_fields] + [
            "karakteristikker",
            "begrunnelse",
        ]
        source_headers = st.session_state.get("last_source_headers", []) or []
        source_keys = [k for k in source_headers if k in ordered_keys]
        fieldnames: List[str] = []

        def _extend(keys: List[str]):
            for key in keys:
                if not key:
                    continue
                if key not in fieldnames:
                    fieldnames.append(key)

        _extend(analysis_order)
        _extend(source_keys)
        remaining = [k for k in ordered_keys if k not in fieldnames]
        _extend(remaining)
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for o in all_rows:
            o2 = {k: o.get(k, "") for k in fieldnames}
            if isinstance(o2.get("karakteristikker"), list):
                o2["karakteristikker"] = "|".join(o2["karakteristikker"])
            for field in category_fields:
                if field["mode"] == "list":
                    values = o2.get(field["key"])
                    if isinstance(values, list):
                        o2[field["key"]] = "|".join(values)
            writer.writerow({k: o2.get(k, "") for k in fieldnames})
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        st.success("Kjøring ferdig ✅")
        if run_mode == "sample":
            st.info("Dette var et sample – bruk «Kjør alt» for å prosessere alle rader.")
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
