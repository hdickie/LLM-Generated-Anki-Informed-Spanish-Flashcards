#https://cvc.cervantes.es/ensenanza/biblioteca_ele/plan_curricular/indice.htm
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Optional, Tuple, Dict, Any, Set
from collections import Counter
import requests
import sqlite3
import json
import os
import re
from datetime import datetime, date, timedelta
from typing import Optional, Literal, List, Dict, Any, Tuple
import time
from dataclasses import dataclass
import spacy
import nltk
from nltk.corpus import wordnet as wn
import langid
import pandas as pd
from functools import lru_cache
import genanki
import math
import wordfreq
from wordfreq import top_n_list
from wordfreq import zipf_frequency
import pprint

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Optional, Set, Tuple, List
import unicodedata



Lang = Literal["es", "en", "amb"]

SPANISH_CHARS = re.compile(r"[áéíóúñü¿¡]", re.IGNORECASE)

SPANISH_FUNCTION_WORDS = {
    "el","la","los","las","un","una","unos","unas",
    "de","del","al","y","o","que","en","por","para",
    "con","sin","a","se","su","sus","mi","mis","tu","tus",
    "es","son","fue","era","ser","estar","hay","no","sí",
}

ENGLISH_FUNCTION_WORDS = {
    "the","a","an","and","or","that","in","on","of","to","for","with","without",
    "is","are","was","were","be","been","it","this","these","those","not","yes","no",
}


from dataclasses import dataclass
from typing import Iterable, List, Optional

import spacy
import nltk
from nltk.corpus import wordnet as wn


EPOCH = date(1970, 1, 1)

BASE_URL = "https://ai.dpc.casa/v1"   # or "http://ai.dpc.casa:8000/v1"
MODEL = "gemma3"                      # whatever name he gave the model
API_KEY = ""                          # fill if he set one, else leave ""

import requests
from typing import Any, Dict, List

ANKI_CONNECT_URL = "http://127.0.0.1:8765"

@lru_cache(maxsize=200_000)
def omw_has_spanish_lemma(word: str) -> bool:
    # Spanish OMW: synsets that contain this word in Spanish
    return len(wn.synsets(word, lang="spa")) > 0

@lru_cache(maxsize=200_000)
def wn_has_english_lemma(word: str) -> bool:
    # English WordNet: synsets for this surface form
    return len(wn.synsets(word)) > 0

@lru_cache(maxsize=200_000)
def looks_like_english(word: str) -> bool:
    # crude but helpful: common English morphology; adjust to your domain
    return bool(re.search(r"(tion|sion|ment|ness|ship|able|ible|ing|ed)$", word))

@lru_cache(maxsize=200_000)
def looks_like_spanish(word: str) -> bool:
    # crude Spanish morphology; adjust to your domain
    return bool(re.search(r"(ción|sión|mente|idad|ismo|ista|oso|osa|amiento|imiento|ando|iendo)$", word))

def top_n_lemmas_with_pos(n: int, nlp):
    words = top_n_list("es", n)
    seen = set()
    out = []

    for w in words:
        doc = nlp(w)
        if not doc:
            continue

        t = doc[0]
        lemma = t.lemma_.lower()
        pos = t.pos_

        key = (lemma, pos)
        if key in seen:
            continue

        seen.add(key)
        out.append((lemma, pos))

    return out

FSRS_API = "http://127.0.0.1:8787/fsrs"

def fetch_fsrs(cids: Iterable[int]) -> list[dict]:
    # batch in chunks so URLs don't get huge
    CHUNK = 200
    cids = list(cids)
    out: list[dict] = []

    for i in range(0, len(cids), CHUNK):
        chunk = cids[i:i+CHUNK]
        params = {"cids": ",".join(map(str, chunk))}
        r = requests.get(FSRS_API, params=params, timeout=5)
        r.raise_for_status()
        out.extend(r.json()["result"])

    return out

def guess_lang_token(token: str, *, prefer: Lang = "None") -> Lang:
    """
    Language guess optimized for single-word vocab lists.
    prefer="es" means: if ambiguous, lean Spanish.
    """
    t = token.strip().lower()
    if not t:
        return "amb"

    # Strong orthographic Spanish signal
    if SPANISH_CHARS.search(t):
        return "es"

    # Function words are high signal
    if t in SPANISH_FUNCTION_WORDS:
        return "es"
    if t in ENGLISH_FUNCTION_WORDS:
        return "en"

    # single-letter or short tokens are genuinely ambiguous
    if len(t) <= 2:
        return "amb"

    # Inventory checks (the big improvement)
    es_in_omw = omw_has_spanish_lemma(t)
    en_in_wn  = wn_has_english_lemma(t)

    # If only one side matches, easy
    if es_in_omw and not en_in_wn:
        return "es"
    if en_in_wn and not es_in_omw:
        return "en"

    # If both match or neither match, score it
    score_es = 0
    score_en = 0

    if es_in_omw: score_es += 3
    if en_in_wn:  score_en += 3

    # morphology nudges
    if looks_like_spanish(t): score_es += 1
    if looks_like_english(t): score_en += 1

    # very light heuristic: Spanish often ends with a vowel, English less so (NOT reliable, just a nudge)
    if t[-1] in "aeiou": score_es += 0  # keep neutral; change to +1 if you want stronger bias

    if score_es > score_en:
        return "es"
    if score_en > score_es:
        return "en"

    # Tie / unknown
    return prefer if prefer in ("es", "en") else "amb"

def ensure_wordnet_downloaded() -> None:
    """
    NLTK ships WordNet separately. OMW data is included under 'omw-1.4' in NLTK.
    """
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)

@dataclass(frozen=True)
class LemmaHit:
    lemma: str
    pos: str
    synset: str          # e.g. "dog.n.01"
    definition_en: str   # WordNet gloss is in English
    examples_en: List[str]
    spanish_lemmas: List[str]

class SpanishLemmaWordnet:
    def __init__(self, spacy_model: str = "es_core_news_sm") -> None:
        ensure_wordnet_downloaded()
        self.nlp = spacy.load(spacy_model, disable=["ner"])  # faster, you can enable if needed

    @staticmethod
    def _spacy_pos_to_wn_pos(spacy_pos: str) -> Optional[str]:
        """
        Map spaCy coarse POS -> WordNet POS tags.
        WordNet supports: n, v, a, r (noun/verb/adj/adv)
        """
        return {
            "NOUN": "n",
            "PROPN": "n",
            "VERB": "v",
            "AUX": "v",
            "ADJ": "a",
            "ADV": "r",
        }.get(spacy_pos)

    def lemmatize(self, text: str) -> List[tuple[str, str, str]]:
        """
        Returns list of (surface, lemma, spacy_pos) excluding punctuation/spaces.
        """
        doc = self.nlp(text)
        out: List[tuple[str, str, str]] = []
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            out.append((t.text, t.lemma_.lower(), t.pos_))
        return out

    def omw_lookup(
        self,
        lemma: str,
        spacy_pos: Optional[str] = None,
        max_synsets: int = 10,
    ) -> List[LemmaHit]:
        """
        Look up a Spanish lemma in WordNet/OMW.
        Note: definitions/examples come from WordNet (English). Spanish lemmas are from OMW.
        """
        wn_pos = self._spacy_pos_to_wn_pos(spacy_pos) if spacy_pos else None

        # wn.synsets(..., lang="spa") queries synsets that have Spanish lemmas in OMW.
        synsets = wn.synsets(lemma, pos=wn_pos, lang="spa")[:max_synsets]

        hits: List[LemmaHit] = []
        for s in synsets:
            spanish_lemmas = sorted({l.name().replace("_", " ") for l in s.lemmas(lang="spa")})
            hits.append(
                LemmaHit(
                    lemma=lemma,
                    pos=s.pos(),
                    synset=s.name(),
                    definition_en=s.definition(),
                    examples_en=list(s.examples()),
                    spanish_lemmas=spanish_lemmas,
                )
            )
        return hits
    
    def analyze_text(self, text: str, max_synsets_per_token: int = 3) -> List[dict]:
        """
        Convenience: lemmatize text, then attach OMW hits per token.
        """
        rows = []
        for surface, lemma, spacy_pos in self.lemmatize(text):
            hits = self.omw_lookup(lemma, spacy_pos=spacy_pos, max_synsets=max_synsets_per_token)
            rows.append(
                {
                    "surface": surface,
                    "lemma": lemma,
                    "pos": spacy_pos,
                    "wn_hits": hits,
                }
            )
        return rows


def anki(action: str, **params) -> Any:
    payload = {"action": action, "version": 6, "params": params}
    r = requests.post(ANKI_CONNECT_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data["result"]

def load_decks(cur) -> Dict[int, str]:
    return {int(did): name for did, name in cur.execute("SELECT id, name FROM decks")}

def load_decks(cur):
    # Deck names use \x1f as hierarchy separator in your DB; convert to Anki-style "::"
    return {
        int(did): (name.replace("\x1f", "::") if isinstance(name, str) else str(name))
        for did, name in cur.execute("SELECT id, name FROM decks")
    }

def table_columns(cur, table):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]

def load_fieldmap_from_fields_table(cur):
    """
    Build mapping: mid -> list of field names ordered by ord
    Uses the `fields` table (which exists in your DB).
    """
    cols = table_columns(cur, "fields")

    # Guess column names across Anki variants
    def pick(*candidates):
        for c in candidates:
            if c in cols:
                return c
        raise RuntimeError(f"Couldn't find any of {candidates} in fields columns={cols}")

    mid_col  = pick("ntid", "mid", "notetype_id", "model_id")
    ord_col  = pick("ord", "idx", "field_ord")
    name_col = pick("name", "fname", "field_name")

    q = f"SELECT {mid_col}, {ord_col}, {name_col} FROM fields ORDER BY {mid_col}, {ord_col}"
    fieldmap = {}
    for mid, ord_, name in cur.execute(q):
        mid = int(mid)
        ord_ = int(ord_)
        fieldmap.setdefault(mid, [])
        # Ensure list is long enough
        while len(fieldmap[mid]) <= ord_:
            fieldmap[mid].append(None)
        fieldmap[mid][ord_] = name

    # Replace None gaps with empty string
    for mid in list(fieldmap.keys()):
        fieldmap[mid] = [x or "" for x in fieldmap[mid]]

    return fieldmap

def pick_front_back_from_fieldmap(flds_str, fieldnames):
    vals = flds_str.split("\x1f")
    lower_names = [n.strip().lower() for n in fieldnames]

    def get_by_name(wanted, fallback_idx):
        if wanted in lower_names:
            i = lower_names.index(wanted)
            return vals[i] if i < len(vals) else ""
        return vals[fallback_idx] if fallback_idx < len(vals) else ""

    front = get_by_name("front", 0)
    back  = get_by_name("back", 1)
    return front, back

def classify(queue, ivl):
    # queue: 0=new, 1=learning, 2=review, 3=relearning, -1=suspended, -2=buried
    if queue == 0:
        return "new"
    if queue == 1:
        return "learning"
    if queue == 3:
        return "relearning"
    if queue == 2:
        return "mature" if (ivl or 0) >= 21 else "review"
    if queue == -1:
        return "suspended"
    if queue == -2:
        return "buried"
    return f"unknown(queue={queue})"

def crt_to_days(col_crt: int) -> int:
    """
    Anki stores col.crt differently across versions.
    Commonly it's a Unix timestamp in seconds (e.g., ~1.7e9).
    If it's already days, it'll be a much smaller number (e.g., ~20k).
    """
    col_crt = int(col_crt)
    # Heuristic: anything bigger than ~100k is almost certainly seconds.
    return col_crt // 86400 if col_crt > 100_000 else col_crt

def due_display(queue: int, due: int, col_crt: int) -> str:
    """
    Best-effort calendar date for review cards.
    In your DB: review due behaves like an offset in *days* from collection creation day.
    """
    due = int(due)
    if queue != 2:
        return f"raw:{due}"

    crt_days = crt_to_days(col_crt)

    # Heuristic: if due is enormous, treat as absolute epoch-day; else treat as crt-relative.
    if due > 100_000:  # ~273 years worth of days, not realistic as a relative due
        day_number = due
    else:
        day_number = crt_days + due

    return (EPOCH + timedelta(days=day_number)).isoformat()



def add_due_fields(df: pd.DataFrame, col_crt: int) -> pd.DataFrame:
    now_ts = int(time.time())
    today_day = (now_ts - int(col_crt)) // 86400

    q = df["queue"].astype(int)
    due = df["due"].astype(int)

    is_active_due = q.isin([1, 2, 3])

    due_in_days = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")

    # Review cards: due is day index
    mask_review = (q == 2)
    due_in_days.loc[mask_review] = (due.loc[mask_review] - today_day)

    # Learning / relearning cards: due is timestamp
    mask_learn = q.isin([1, 3])
    due_in_days.loc[mask_learn] = (due.loc[mask_learn] - now_ts) / 86400.0

    df = df.copy()
    df["is_active_due"] = is_active_due
    df["due_in_days"] = due_in_days

    return df


def get_anki_connection(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con

def get_cards_notes_df(con: sqlite3.Connection) -> pd.DataFrame:
    rows = con.execute("""
        SELECT
            c.id   AS card_id,
            c.nid  AS note_id,
            c.did  AS deck_id,
            c.queue,
            c.type,
            c.due,
            c.ivl,
            c.reps,
            c.lapses,
            c.factor,
            c.left,
            c.odue,
            c.odid,
            n.flds,
            n.tags,
            n.mid AS model_id
        FROM cards c
        JOIN notes n ON n.id = c.nid
    """).fetchall()
    return pd.DataFrame([dict(r) for r in rows])

def get_revlog_df(con: sqlite3.Connection) -> pd.DataFrame:
    # revlog schema (Anki 2.1+): id, cid, usn, ease, ivl, lastIvl, factor, time, type

    deck_root_ui = "Espanol::Active Learning::Dave LLM Sentences"
    deck_root_us = deck_root_ui.replace("::", chr(31))  # '\x1f'

    rows = con.execute("""
        SELECT
            -- revlog (event-level)
            r.id        AS review_id,
            r.cid       AS card_id,
            r.ivl       AS ivl_after,
            r.lastIvl   AS ivl_before,
            r.time      AS time_ms,
            r.type      AS review_type,
            r.ease      AS ease,

            -- cards (snapshot)
            c.nid       AS note_id,
            c.did       AS deck_id,
            c.queue     AS queue,
            c.type      AS type,
            c.due       AS due,
            c.ivl       AS ivl,
            c.reps      AS reps,
            c.lapses    AS lapses,
            c.left      AS left,
            c.odue      AS odue,
            c.odid      AS odid,

            -- notes (content)
            n.flds      AS flds,
            n.tags      AS tags,
            n.mid       AS model_id,

            -- deck (debug / context)
            CAST(d.name AS TEXT) AS deck_name

        FROM revlog r
        JOIN cards c ON r.cid = c.id
        JOIN notes n ON c.nid = n.id
        JOIN decks d ON c.did = d.id

        WHERE CAST(d.name AS TEXT) COLLATE BINARY = ?
        OR CAST(d.name AS TEXT) COLLATE BINARY LIKE ? || char(31) || '%'

        ORDER BY r.id;
    """, (deck_root_us, deck_root_us)).fetchall()


    df = pd.DataFrame([dict(r) for r in rows])

    # df_first_review = pd.read_sql_query("""
    #     SELECT
    #         CAST(d.name AS TEXT) AS deck_name,
    #         MIN(r.id) AS first_review_id_ms
    #     FROM revlog r
    #     JOIN cards c ON r.cid = c.id
    #     JOIN decks d ON c.did = d.id
    #     WHERE CAST(d.name AS TEXT) COLLATE BINARY = ?
    #     OR CAST(d.name AS TEXT) COLLATE BINARY LIKE ? || char(31) || '%'
    #     GROUP BY deck_name COLLATE BINARY
    #     ORDER BY deck_name COLLATE BINARY;
    # """, con, params=(deck_root_us, deck_root_us))

    # df_first_review["first_review_dt"] = (
    #     pd.to_datetime(df_first_review["first_review_id_ms"], unit="ms", utc=True)
    #     .dt.tz_convert("America/Los_Angeles")
    # )

    df = df[df["note_id"].notna()] #drop rows for reviews of cards which no longer exist

    # review_id is ms since epoch
    df["review_dt"] = pd.to_datetime(df["review_id"], unit="ms", utc=True).dt.tz_convert("America/Los_Angeles")
    df["time_s"] = df["time_ms"] / 1000.0

    BUTTON_MAP = {
        1: "again",
        2: "hard",
        3: "good",
        4: "easy",
    }

    df["button"] = df["ease"].map(BUTTON_MAP)

    return df

# def join_revlog_cards(con: sqlite3.Connection) -> pd.DataFrame:
#     cards = get_cards_notes_df(con)
#     rev = get_revlog_df(con)

#     df = rev.merge(cards, on="card_id", how="left", validate="many_to_one")

#     # ease mapping:
#     # 1=Again, 2=Hard, 3=Good, 4=Easy
#     ease_map = {1: "again", 2: "hard", 3: "good", 4: "easy"}
#     df["button"] = df["ease"].map(ease_map).fillna("unknown")

#     return df


def getAnkiCards():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    decks = load_decks(cur)
    fieldmap = load_fieldmap_from_fields_table(cur)
    col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

    rows = cur.execute("""
        SELECT
            c.id   AS card_id,
            c.nid  AS note_id,
            c.did  AS deck_id,
            c.queue,
            c.type,
            c.due,
            c.ivl,
            c.reps,
            c.lapses,
            c.factor,
            c.left,
            c.odue,
            c.odid,

            n.flds,
            n.tags,
            n.mid AS model_id
        FROM cards c
        JOIN notes n ON n.id = c.nid
        ORDER BY c.did, c.id
    """).fetchall()

    card_data = []
    for r in rows:
        model_id = int(r["model_id"])
        fieldnames = fieldmap.get(model_id, [])
        front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

        raw_tags = (r["tags"] or "").strip()
        tags = raw_tags.split() if raw_tags else []

        deck_name = decks.get(int(r["deck_id"]), f"Unknown(did={r['deck_id']})")
        if not deck_name.startswith("Espanol::Active Learning"):
            continue

        noun_seeds = (
            [t[len("nounseed:"):] for t in tags if t.startswith("nounseed:")]
            + [t[len("auto-lemma-noun:"):] for t in tags if t.startswith("auto-lemma-noun:")]
        )

        verb_seeds = (
            [t[len("verbseed:"):] for t in tags if t.startswith("verbseed:")]
            + [t[len("auto-lemma-verb:"):] for t in tags if t.startswith("auto-lemma-verb:")]
        )


        out = {
            # Identity / joins
            "card_id": int(r["card_id"]),
            "note_id": int(r["note_id"]),
            "deck_id": int(r["deck_id"]),
            "deck": deck_name,
            "model_id": model_id,

            # Content
            "front": front,
            "back": back,
            "tags": tags,
            "raw_tags": raw_tags,

            # Scheduling (raw)
            "queue": int(r["queue"]),
            "type": int(r["type"]),
            "due": int(r["due"]),
            "ivl": int(r["ivl"] or 0),
            "reps": int(r["reps"] or 0),
            "lapses": int(r["lapses"] or 0),
            "factor": int(r["factor"] or 0),
            "left": int(r["left"] or 0),
            "odue": int(r["odue"] or 0),
            "odid": int(r["odid"] or 0),

            # Your existing derived fields
            "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
            "due_display": due_display(int(r["queue"]), int(r["due"]), col_crt),

            # Seed extraction helpers
            "noun_seeds": noun_seeds,   # list (usually length 0 or 1)
            "verb_seeds": verb_seeds,   # list (usually length 0 or 1)
            "has_nounseed": any(t.startswith("nounseed:") for t in tags),
            "has_verbseed": any(t.startswith("verbseed:") for t in tags),

            # Optional: quick label to filter “sentence cards” by tag conventions
            # (adjust this to your real tag/note-type convention)
            "is_sentence_card": ("SENTENCE" in tags) or ("sentence" in tags) or ("Sentence" in tags),
        }

        card_data.append(out)

    card_data_df = pd.DataFrame(card_data)
    card_data_df = add_due_fields(card_data_df, col_crt)

    # print(card_data_df)
    return card_data_df

def getAnkiSentenceCards():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    decks = load_decks(cur)
    fieldmap = load_fieldmap_from_fields_table(cur)
    col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

    rows = cur.execute("""
        SELECT
            c.id   AS card_id,
            c.nid  AS note_id,
            c.did  AS deck_id,
            c.queue,
            c.type,
            c.due,
            c.ivl,
            c.reps,
            c.lapses,
            c.factor,
            c.left,
            c.odue,
            c.odid,

            n.flds,
            n.tags,
            n.mid AS model_id
        FROM cards c
        JOIN notes n ON n.id = c.nid
        ORDER BY c.did, c.id
    """).fetchall()

    card_data = []
    for r in rows:
        # print(dict(r))
        model_id = int(r["model_id"])
        fieldnames = fieldmap.get(model_id, [])
        front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

        raw_tags = (r["tags"] or "").strip()
        tags = raw_tags.split() if raw_tags else []

        deck_name = decks.get(int(r["deck_id"]), f"Unknown(did={r['deck_id']})")
        if not deck_name.startswith("Espanol::Active Learning::Dave LLM Sentences"):
            continue

        # print(dict(r))

        # Seeds pulled directly from tags (you can parse later too; this is convenient)
        noun_seeds = [t[len("nounseed:"):] for t in tags if t.startswith("nounseed:")]
        verb_seeds = [t[len("verbseed:"):] for t in tags if t.startswith("verbseed:")]

        out = {
            # Identity / joins
            "card_id": int(r["card_id"]),
            "note_id": int(r["note_id"]),
            "deck_id": int(r["deck_id"]),
            "deck": deck_name,
            "model_id": model_id,

            # Content
            "front": front,
            "back": back,
            "tags": tags,
            "raw_tags": raw_tags,

            # Scheduling (raw)
            "queue": int(r["queue"]),
            "type": int(r["type"]),
            "due": int(r["due"]),
            "ivl": int(r["ivl"] or 0),
            "reps": int(r["reps"] or 0),
            "lapses": int(r["lapses"] or 0),
            "factor": int(r["factor"] or 0),
            "left": int(r["left"] or 0),
            "odue": int(r["odue"] or 0),
            "odid": int(r["odid"] or 0),

            # Your existing derived fields
            "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
            "due_display": due_display(int(r["queue"]), int(r["due"]), col_crt),

            # Seed extraction helpers
            "noun_seeds": noun_seeds,   # list (usually length 0 or 1)
            "verb_seeds": verb_seeds,   # list (usually length 0 or 1)
            "has_nounseed": any(t.startswith("nounseed:") for t in tags),
            "has_verbseed": any(t.startswith("verbseed:") for t in tags),

            # Optional: quick label to filter “sentence cards” by tag conventions
            # (adjust this to your real tag/note-type convention)
            "is_sentence_card": ("SENTENCE" in tags) or ("sentence" in tags) or ("Sentence" in tags),
        }

        card_data.append(out)

    card_data_df = pd.DataFrame(card_data)
    card_data_df = add_due_fields(card_data_df, col_crt)

    # print(card_data_df)
    return card_data_df


# def main():
#     con = sqlite3.connect(DB_PATH)
#     con.row_factory = sqlite3.Row
#     cur = con.cursor()

#     decks = load_decks(cur)
#     fieldmap = load_fieldmap_from_fields_table(cur)
#     col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

#     rows = cur.execute("""
#         SELECT
#             c.id AS card_id,
#             c.nid,
#             c.did,
#             c.queue,
#             c.due,
#             c.ivl,
#             n.flds,
#             n.tags,
#             n.mid
#         FROM cards c
#         JOIN notes n ON n.id = c.nid
#         ORDER BY c.did, c.id
#         LIMIT 20
#     """).fetchall()

#     for r in rows:
#         mid = int(r["mid"])
#         fieldnames = fieldmap.get(mid, [])
#         front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

#         out = {
#             "deck": decks.get(int(r["did"]), f"Unknown(did={r['did']})"),
#             "front": front,
#             "back": back,
#             "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
#             "due": due_display(int(r["queue"]), int(r["due"]), col_crt),
#             "tags": (r["tags"] or "").strip().split(),
#         }
#         print(out)

#     con.close()

def append_tag(note_id: int, tag: str) -> None:
    """
    Append a tag to a note if it doesn't already have it.
    """
    # AnkiConnect expects space-separated tags
    anki("addTags", notes=[note_id], tags=tag)

def translate_es_to_en(prompt: str) -> str:
    url = f"{BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL,
        "messages": [
        {
            "role": "system",
            "content": (
                "You are a Spanish-to-English sentence translator. "
                "Respond with exactly ONE complete English sentence. "
                "Do NOT include explanations, translations, punctuation outside the sentence, "
                "or any text in any language other than English."
            )
            # You are a Spanish-to-English sentence translator for language-learning flashcards.
            # Goal: Produce an English sentence that preserves the Spanish sentence’s grammatical structure and information order as closely as possible (clause order, fronted phrases, conditionals, passive/impersonal “se”, etc.), while remaining grammatical English.
            # Rules:
            # - Output EXACTLY ONE complete English sentence.
            # - Do NOT add explanations or any extra text.
            # - Keep the same clause order as the Spanish whenever possible (e.g., if the Spanish starts with a conditional phrase like “A condición de que…”, the English should also start with an equivalent conditional phrase).
            # - Preserve voice: if Spanish uses passive/impersonal (“se + verb”), prefer passive/impersonal English (“is/are …”, “one/they …”) rather than switching to an active paraphrase, unless impossible.
            # - Prefer literal/structural faithfulness over stylistic naturalness.
            # - Keep meaning intact; do not omit details.

        },
        {"role": "user", "content": prompt}
    ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Standard OpenAI-style response shape
    return data["choices"][0]["message"]["content"]


def chat(prompt: str) -> str:
    url = f"{BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL,
        "messages": [
        {
            "role": "system",
            "content": (
                # "You are a Spanish sentence generator. "
                # "Respond with exactly ONE complete Spanish sentence. "
                # "Do NOT include explanations, translations, punctuation outside the sentence, "
                # "or any text in any language other than Spanish."
                # "Use the exact grammatical construction or connector requested."
                # "Do not paraphrase or substitute it."
                # "The sentence must demonstrate the requested grammatical reason clearly and unambiguously."
                """
                You are a Spanish sentence generator.
                Respond with exactly ONE complete Spanish sentence.
                Use the exact grammatical construction or connector requested.
                Do not paraphrase, substitute, or restructure to avoid the construction.
                The sentence must clearly and unambiguously demonstrate the requested grammatical reason.
                Do NOT include explanations, translations, punctuation outside the sentence,
                or any text in any language other than Spanish.
                """
            )
            # "content": (
            #     "You are a part of speech determiner for both English and Spanish. "
            #     "Respond with exactly ONE part of speech: noun, verb, adjective, adverb, other "
            #     "Do NOT include explanations, translations, punctuation outside the sentence. "
            

            # )
        },
        {"role": "user", "content": prompt}
    ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Standard OpenAI-style response shape
    return data["choices"][0]["message"]["content"]

def generate_spanish_word_definition(prompt: str) -> str:
    url = f"{BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL,
        "messages": [
        {
            "role": "system",
            "content": (
                # "You are a Spanish sentence generator. "
                # "Respond with exactly ONE complete Spanish sentence. "
                # "Do NOT include explanations, translations, punctuation outside the sentence, "
                # "or any text in any language other than Spanish."
                # "Use the exact grammatical construction or connector requested."
                # "Do not paraphrase or substitute it."
                # "The sentence must demonstrate the requested grammatical reason clearly and unambiguously."
                """
                You are a Spanish lexicography assistant.

                Given a single Spanish word or short phrase as input. 
                The input may include an article (el, la, los, las, un, una, etc.) and may not be in lemma form.
                Respond with exactly ONE concise definition in English suitable for the back of an Anki card.

                Rules:
                - Be brief and concrete (aim for one short sentence or a compact phrase).
                - Capture the most common, neutral meaning.
                - Interpret inflected or conjugated forms as their base meaning and respond with the unconjugated sense.
                - Do NOT include examples, translations lists, usage notes, etymology, or regional commentary.
                - Do NOT include articles unless they are essential to meaning.
                - Do NOT include punctuation other than commas if strictly necessary.
                - Do NOT include quotes, parentheses, emojis, or formatting.
                - Do NOT mention parts of speech explicitly.
                - Do NOT output the word itself.
                - Output English only.
                """
            )
            # "content": (
            #     "You are a part of speech determiner for both English and Spanish. "
            #     "Respond with exactly ONE part of speech: noun, verb, adjective, adverb, other "
            #     "Do NOT include explanations, translations, punctuation outside the sentence. "
            

            # )
        },
        {"role": "user", "content": prompt}
    ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Standard OpenAI-style response shape
    return data["choices"][0]["message"]["content"]

def build_anki_deck(deck_name, deck_dict):
    # e.g.
    # deck = {
    #     "hola": {"back": "hello", "tags": ["spanish", "greeting"]},
    #     "adiós": ("goodbye", ["spanish"]),
    #     "gracias": "thanks",  # no tags still works
    # }
    my_deck = genanki.Deck(
        hash(deck_name),
        deck_name
    )

    my_model = genanki.Model(
        hash('Simple Front-Back Card'),
        'Simple Front-Back Card',
        fields=[
            {'name': 'Front'},
            {'name': 'Back'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Front}}',
                'afmt': '{{Front}}<hr id="answer">{{Back}}',
            },
        ],
    )

    for front, spec in deck_dict.items():
        back = None
        tags = []

        # Backwards-compatible: {"front": "back"}
        if isinstance(spec, str):
            back = spec

        # Tuple form: ("back", ["tag1", "tag2"])
        elif isinstance(spec, tuple) and len(spec) == 2:
            back, tags = spec
            tags = list(tags) if tags else []

        # Dict form: {"back": "...", "tags": [...]}
        elif isinstance(spec, dict):
            back = spec.get("back")
            tags = list(spec.get("tags", []))

        else:
            raise TypeError(
                f"Unsupported card spec for front={front!r}: {type(spec).__name__}"
            )

        if back is None:
            raise ValueError(f"Missing 'back' for front={front!r}")

        tags = [s.replace(" ", "-") for s in tags]

        my_note = genanki.Note(
            model=my_model,
            fields=[front, back],
            tags=tags,  # <-- HERE
        )
        my_deck.add_note(my_note)

    return my_deck
    # genanki.Package(my_deck).write_to_file(deck_file_name + ".apkg")

import numpy as np
import pandas as pd

# class SeedPicker:
#     def __init__(self, usage_stats: pd.DataFrame, pos: str, rng_seed: int = 0):
#         print(usage_stats)
#         self.df = usage_stats[usage_stats["part_of_speech"] == pos].copy()
#         self.rng = np.random.default_rng(rng_seed)

#         # Ensure missing columns exist
#         for col in ["new_count", "young_count", "mature_count", "total_count"]:
#             if col not in self.df.columns:
#                 self.df[col] = 0

#     def _score(self, row) -> float:
#         # Tune these weights to match your pain (new/young drives workload)
#         return (
#             row["total_count"]
#             + 3.0 * row["new_count"]
#             + 2.0 * row["young_count"]
#             + 0.5 * row["mature_count"]
#         )

#     def next(self) -> str:
#         # Compute scores
#         scores = self.df.apply(self._score, axis=1).to_numpy()

#         # Pick from the best K to keep variety
#         K = min(50, len(self.df))
#         best_idx = np.argpartition(scores, K - 1)[:K]

#         # Convert score -> sampling weight (lower score => higher weight)
#         # Add epsilon to avoid division by 0.
#         eps = 1e-6
#         w = 1.0 / (scores[best_idx] + eps)
#         w = w / w.sum()

#         chosen_local = self.rng.choice(best_idx, p=w)
#         word = self.df.iloc[chosen_local]["word"]

#         # Optimistically account for the card you’re about to create
#         self.df.iloc[chosen_local, self.df.columns.get_loc("new_count")] += 1
#         self.df.iloc[chosen_local, self.df.columns.get_loc("total_count")] += 1

#         return word

def _num(x, default=0.0):
    """Convert possible pd.NA / NaN to a real float."""
    try:
        if x is None:
            return default
        if x is pd.NA:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default

class SeedPicker:
    def __init__(self, usage_stats, rng_seed=0):
        self.df = usage_stats.copy()
        self.rng = np.random.default_rng(rng_seed)

        # These are the only ones you should be mutating during generation,
        # since you're creating *sentence* cards that *use* the seed.
        required_sent_cols = [
            "sent_total_count",
            "sent_new_count",
            "sent_young_count",
            "sent_mature_count",
            "sent_learning_count",
            "sent_suspended_count",
            "sent_buried_count",
            "sent_load_score",
            "sent_to_seed_ratio",
        ]
        for col in required_sent_cols:
            if col not in self.df.columns:
                # counts are numeric; use float to match your current DF (0.0)
                self.df[col] = 0.0

        # Seed-side columns are read-only for picking; ensure presence only if you rely on them in _score
        required_seed_cols = [
            "seed_total_count",
            "seed_reps_sum",
            "seed_mature_count",
            "seed_lapse_rate",
            "seed_near_term_due_pressure",
        ]
        for col in required_seed_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0

        # Optional: drop legacy columns if they exist, to avoid accidental use
        legacy = ["new_count", "young_count", "mature_count", "total_count"]
        for col in legacy:
            if col in self.df.columns:
                # safer than deleting if other code expects them:
                # set to 0 and stop relying on them
                self.df[col] = 0.0


    def _score(self, row):
        """
        Lower score = better candidate
        """

        # ---- HARD EXCLUSIONS ----
        if _num(row.get("seed_suspended_count")) > 0:
            return float("inf")
        if _num(row.get("seed_buried_count")) > 0:
            return float("inf")

        reps = _num(row.get("seed_reps_sum"))
        if reps < 3:
            return float("inf")

        lapse_rate = _num(row.get("seed_lapse_rate"))
        if lapse_rate > 0.25:
            return float("inf")

        # ---- REUSE PENALTY ----
        sent_uses = _num(row.get("sent_total_count"))
        reuse_penalty = sent_uses ** 2

        # ---- FAMILIARITY BONUS ----
        mature = _num(row.get("seed_mature_count"))
        familiarity_bonus = math.log1p(reps) + 0.5 * mature

        # ---- REVIEW LOAD PENALTY ----
        near_due = _num(row.get("seed_near_term_due_pressure"))
        load_penalty = 0.5 * near_due + 2.0 * lapse_rate

        # ---- FINAL SCORE ----
        score = (
            2.5 * reuse_penalty
            + 1.0 * load_penalty
            - 3.0 * familiarity_bonus
        )

        return score

    def next(self):
        # Compute scores (assumes: LOWER is better)
        scores = self.df.apply(self._score, axis=1).to_numpy(dtype=float)

        n = len(self.df)
        if n == 0:
            raise ValueError("No candidates left in df")

        K = min(50, n)
        best_idx = np.argpartition(scores, K - 1)[:K]  # indices of K smallest scores

        # ---- Sampling among top-K with a little randomness ----
        # We want higher weight for smaller scores. Make it robust to:
        #  - zeros
        #  - negatives
        #  - NaNs / inf
        top_scores = scores[best_idx].copy()

        # Replace non-finite with a large number (bad)
        bad = ~np.isfinite(top_scores)
        if bad.any():
            top_scores[bad] = np.nanmax(top_scores[~bad]) if (~bad).any() else 0.0
            top_scores[bad] = top_scores[bad] + 1.0

        # Shift so the minimum is >= eps (handles negatives)
        eps = 1e-6
        min_s = float(np.min(top_scores))
        shifted = top_scores - min_s + eps

        weights = 1.0 / shifted
        wsum = float(weights.sum())
        if not np.isfinite(wsum) or wsum <= 0:
            # fallback: uniform
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= wsum

        choice_idx = int(self.rng.choice(best_idx, p=weights))
        word = self.df.iloc[choice_idx]["word"]

        # ---- optimistic update: we just created a NEW sentence card using this seed ----
        row_ix = self.df.index[choice_idx]

        # Ensure columns exist (in case you run with partial schemas)
        for col in ["sent_total_count", "sent_new_count", "sent_load_score", "sent_to_seed_ratio",
                    "seed_total_count"]:
            if col not in self.df.columns:
                # Only create what we need; defaults match your stats meaning
                self.df[col] = 0.0 if col.startswith("sent_") else 0

        # New sentence card using this seed:
        self.df.at[row_ix, "sent_total_count"] = float(self.df.at[row_ix, "sent_total_count"]) + 1.0
        self.df.at[row_ix, "sent_new_count"]   = float(self.df.at[row_ix, "sent_new_count"]) + 1.0

        # Keep sent_load_score consistent with your stats formula:
        # load_score = total + 3*new + 2*young + 0.5*mature
        # Here we only bumped total+new by 1
        self.df.at[row_ix, "sent_load_score"] = float(self.df.at[row_ix, "sent_load_score"]) + (1.0 + 3.0)

        # Update ratio (usually seed_total_count==1, but keep general)
        seed_total = float(self.df.at[row_ix, "seed_total_count"]) if pd.notna(self.df.at[row_ix, "seed_total_count"]) else 0.0
        sent_total = float(self.df.at[row_ix, "sent_total_count"])
        self.df.at[row_ix, "sent_to_seed_ratio"] = 0.0 if seed_total <= 0 else (sent_total / seed_total)

        return word

from typing import Iterator, Tuple
def make_generators(noun_usage_stats: pd.DataFrame,
                    verb_usage_stats: pd.DataFrame,
                    rng_seed: int = 42) -> Iterator[Tuple[str, str]]:
    """
    Returns an infinite generator yielding (noun, verb) pairs, biased toward
    least-used / lowest-review-load seeds.

    Expects each DF to have at least:
      - 'word'
    And optionally:
      - 'new_count', 'young_count', 'mature_count', 'total_count'
    """

    noun_picker = SeedPicker(noun_usage_stats, rng_seed=rng_seed)
    verb_picker = SeedPicker(verb_usage_stats, rng_seed=rng_seed + 1)

    while True:
        yield noun_picker.next(), verb_picker.next()

MATURE_IVL_DAYS_DEFAULT = 21  # tune if you want (Anki “mature” is commonly 21d+)

def extractSeedStatistics(
    cards_df: pd.DataFrame,
    *,
    mature_ivl_days: int = MATURE_IVL_DAYS_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    (noun_usage_stats_df, verb_usage_stats_df)

    Output columns include BOTH:
    - seed_* : stats from the seed's own POS card(s) (posid:*, langid:es)
    - sent_* : stats from sentence cards that USED the seed (noun_seeds/verb_seeds)

    PLUS_CONFIRMATION_KEYS:
    - seed_card_ids : list[int] of card_ids for the seed’s own POS cards (deduped)
    - sent_card_ids : list[int] of card_ids for sentence cards that used the seed (deduped)

    Notes:
    - We do NOT overwrite noun_seeds/verb_seeds. Those should represent sentence usage.
    - If your upstream pipeline already created noun_seeds/verb_seeds/is_sentence_card/etc.,
        this function will respect them.
    """

    df = cards_df.copy()

    # ---------- basic normalization ----------
    if "tags" not in df.columns:
        raise ValueError("cards_df must include 'tags' column (list[str])")

    # front clean
    df["front_seed"] = df["front"].fillna("").astype(str).str.strip()

    # required scheduling fields
    for col in ["queue", "ivl"]:
        if col not in df.columns:
            raise ValueError(f"cards_df is missing required column: '{col}'")
    df["queue"] = df["queue"].fillna(0).astype(int)
    df["ivl"] = df["ivl"].fillna(0).astype(int)

    # ensure optional numeric columns exist
    for col in ["reps", "lapses", "factor"]:
        if col not in df.columns:
            df[col] = 0

    # ---------- helper tag checks ----------
    def _has_tag(ts, tag: str) -> bool:
        return bool(ts) and (tag in ts)

    def _has_prefix(ts, prefix: str) -> bool:
        if not ts:
            return False
        return any(t.startswith(prefix) for t in ts)

    # ---------- classification flags ----------
    # Anki queue reference (common):
    #  0=new, 1=learning, 2=review, 3=day learn, -1=suspended, -2=buried
    df["is_new"]       = (df["queue"] == 0)
    df["is_review"]    = (df["queue"] == 2)
    df["is_learning"]  = df["queue"].isin([1, 3])
    df["is_suspended"] = (df["queue"] == -1)
    df["is_buried"]    = (df["queue"] == -2)

    df["is_mature"] = df["is_review"] & (df["ivl"] >= mature_ivl_days)
    df["is_young"]  = (df["is_review"] & (df["ivl"] < mature_ivl_days)) | df["is_learning"]

    # due fields (optional but you already have them)
    if "is_active_due" not in df.columns:
        # best-effort default: only learning/review are "active"
        df["is_active_due"] = df["queue"].isin([1, 2, 3])
    if "due_in_days" not in df.columns:
        df["due_in_days"] = pd.NA

    # ---------- identify card types ----------
    df["is_pos_noun_card"] = df["tags"].apply(lambda ts: _has_tag(ts, "posid:NOUN"))
    df["is_pos_verb_card"] = df["tags"].apply(lambda ts: _has_tag(ts, "posid:VERB"))
    df["is_lang_es"]       = df["tags"].apply(lambda ts: _has_tag(ts, "langid:es"))

    # sentence usage: prefer your explicit columns if present; otherwise infer from tag prefixes
    if "is_sentence_card" not in df.columns:
        df["is_sentence_card"] = df["tags"].apply(lambda ts: _has_prefix(ts, "learningobjective:"))

    if "noun_seeds" not in df.columns:
        df["noun_seeds"] = [[] for _ in range(len(df))]
    if "verb_seeds" not in df.columns:
        df["verb_seeds"] = [[] for _ in range(len(df))]

    # ---------- aggregator used for both streams ----------
    def _stats_for_seedcol(frame: pd.DataFrame, seed_col: str, *, stream_prefix: str) -> pd.DataFrame:
        """
        Explodes seed_col to (card_id, word) rows, then aggregates per word.
        Also propagates contributing card_ids into a list column:
          f"{stream_prefix}card_ids"
        """
        cols = [
            "card_id", seed_col,
            "is_new", "is_young", "is_mature", "is_learning", "is_suspended", "is_buried",
            "is_active_due", "due_in_days",
            "reps", "lapses", "factor",
        ]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"_stats_for_seedcol missing required columns in df: {missing}")

        tmp = frame[cols].copy()
        tmp = tmp.explode(seed_col, ignore_index=True).rename(columns={seed_col: "word"})
        tmp = tmp[tmp["word"].notna() & (tmp["word"].astype(str).str.len() > 0)]

        # normalize due
        try:
            tmp["due_in_days"] = tmp["due_in_days"].astype("Float64")
        except Exception:
            tmp["due_in_days"] = pd.to_numeric(tmp["due_in_days"], errors="coerce").astype("Float64")

        tmp["due_in_days_active"] = tmp["due_in_days"].where(tmp["is_active_due"] == True)

        # card_ids list aggregator
        card_ids_col = f"{stream_prefix}card_ids"

        out = (
            tmp.groupby("word", as_index=False)
            .agg(
                total_count=("card_id", "count"),
                new_count=("is_new", "sum"),
                young_count=("is_young", "sum"),
                mature_count=("is_mature", "sum"),
                learning_count=("is_learning", "sum"),
                suspended_count=("is_suspended", "sum"),
                buried_count=("is_buried", "sum"),
                next_due_days=("due_in_days_active", "min"),
                due_today_count=("due_in_days_active", lambda s: (s.notna() & (s <= 0) & (s > -1)).sum()),
                overdue_count=("due_in_days_active", lambda s: (s.notna() & (s < 0)).sum()),
                due_7d_count=("due_in_days_active", lambda s: (s.notna() & (s >= 0) & (s <= 7)).sum()),
                reps_sum=("reps", "sum"),
                lapses_sum=("lapses", "sum"),
                avg_factor=("factor", "mean"),
                **{card_ids_col: ("card_id", lambda s: sorted(set(int(x) for x in s.dropna().tolist())))},
            )
        )

        out["near_term_due_pressure"] = out["due_today_count"] + out["due_7d_count"]
        out["lapse_rate"] = (
            out["lapses_sum"] / out["reps_sum"].replace(0, pd.NA)
        ).fillna(0.0).infer_objects(copy=False)


        out["load_score"] = (
            out["total_count"]
            + 3.0 * out["new_count"]
            + 2.0 * out["young_count"]
            + 0.5 * out["mature_count"]
        )
        return out

    # ---------- (A) seed-card stats: the word itself ----------
    noun_seed_cards = df[df["is_pos_noun_card"] & df["is_lang_es"]].copy()
    verb_seed_cards = df[df["is_pos_verb_card"] & df["is_lang_es"]].copy()

    print('Count Cards........: '+str(df.shape[0]))
    print('Count Nouns........: '+str(df["is_pos_noun_card"].sum()))
    print('Count Verbs........: '+str(df["is_pos_verb_card"].sum()))
    print('Count Spanish......: '+str(df["is_lang_es"].sum()))
    print('Count Spanish Nouns: '+str(noun_seed_cards.shape[0]))
    print('Count Spanish Verbs: '+str(verb_seed_cards.shape[0]))

    noun_seed_cards["seed_word"] = noun_seed_cards["front_seed"].apply(lambda s: [s] if s else [])
    verb_seed_cards["seed_word"] = verb_seed_cards["front_seed"].apply(lambda s: [s] if s else [])

    noun_seed_stats = _stats_for_seedcol(
        noun_seed_cards, "seed_word", stream_prefix="seed_"
    ).rename(
        columns=lambda c: ("seed_" + c)
        if c not in {"word", "seed_card_ids"}
        else c
    )
    verb_seed_stats = _stats_for_seedcol(verb_seed_cards, "seed_word", stream_prefix="seed_").rename(
        columns=lambda c: ("seed_" + c) if c not in {"word", "seed_card_ids"} else c
    )

    # ---------- (B) sentence-usage stats: where the word was used as a seed ----------
    sentence_cards = df[df["is_sentence_card"] == True].copy()

    noun_sentence_stats = _stats_for_seedcol(
        sentence_cards, "noun_seeds", stream_prefix="sent_"
    ).rename(
        columns=lambda c: ("sent_" + c)
        if c not in {"word", "sent_card_ids"}
        else c
    )

    verb_sentence_stats = _stats_for_seedcol(sentence_cards, "verb_seeds", stream_prefix="sent_").rename(
        columns=lambda c: ("sent_" + c) if c not in {"word", "sent_card_ids"} else c
    )

    # ---------- merge streams ----------
    noun_usage = noun_seed_stats.merge(noun_sentence_stats, on="word", how="outer")
    verb_usage = verb_seed_stats.merge(verb_sentence_stats, on="word", how="outer")

    # fill the count-ish numeric columns with 0 where missing (leave next_due_days as NA)
    def _fill_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for c in out.columns:
            if c == "word":
                continue
            if c.endswith("next_due_days"):
                continue
            if c.endswith("card_ids"):
                continue  # lists; leave NaN
            if pd.api.types.is_numeric_dtype(out[c]) or out[c].dtype == "Float64":
                out[c] = out[c].fillna(0)
        return out

    noun_usage = _fill_numeric(noun_usage)
    verb_usage = _fill_numeric(verb_usage)

    # ensure id lists are always lists (not NaN)
    for frame in (noun_usage, verb_usage):
        for col in ["seed_card_ids", "sent_card_ids"]:
            if col in frame.columns:
                frame[col] = frame[col].apply(lambda x: x if isinstance(x, list) else [])

    noun_usage["sent_to_seed_ratio"] = (
        noun_usage.get("sent_total_count", 0) / noun_usage.get("seed_total_count", 0).replace(0, pd.NA)
    ).fillna(0.0)
    verb_usage["sent_to_seed_ratio"] = (
        verb_usage.get("sent_total_count", 0) / verb_usage.get("seed_total_count", 0).replace(0, pd.NA)
    ).fillna(0.0)

    noun_usage = noun_usage.sort_values(
        by=["sent_total_count", "seed_load_score", "word"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    verb_usage = verb_usage.sort_values(
        by=["sent_total_count", "seed_load_score", "word"],
        ascending=[True, True, True],
        na_position="last",
        ).reset_index(drop=True)

    return noun_usage, verb_usage

def add_hardcoded_seeds(usage_stats: pd.DataFrame, extra_words: list[str]) -> pd.DataFrame:
    """
    Ensure extra_words exist in usage_stats with 0 counts if unseen.
    Expects usage_stats has column: 'word' and count columns.
    """
    extra = pd.DataFrame({"word": [w.strip() for w in extra_words if w and w.strip()]})
    extra = extra.drop_duplicates()

    # Merge: keep all existing + all extra
    out = pd.concat([usage_stats[["word"]], extra], ignore_index=True).drop_duplicates(subset=["word"])

    # Bring counts back (left join existing counts)
    out = out.merge(usage_stats, on="word", how="left")

    # Fill missing counts with 0
    for col in ["total_count", "new_count", "young_count", "mature_count", "learning_count", "suspended_count", "buried_count"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)

    # Recompute load_score if present/needed
    if set(["total_count", "new_count", "young_count", "mature_count"]).issubset(out.columns):
        out["load_score"] = (
            out["total_count"]
            + 3.0 * out["new_count"]
            + 2.0 * out["young_count"]
            + 0.5 * out["mature_count"]
        )

    return out

MATURE_IVL_DAYS_DEFAULT = 21  # tweak if you want

def extractObjectiveStatistics(
    cards_df: pd.DataFrame,
    *,
    objective_prefix: str = "learningobjective:",
    mature_ivl_days: int = MATURE_IVL_DAYS_DEFAULT,
) -> pd.DataFrame:
    """
    Usage stats per learning objective extracted from tags (learningobjective:...).

    Requires columns in cards_df:
      - tags (list[str] or space-delimited string)
      - queue (int), ivl (int)
    Optional but used if present:
      - card_id
      - due_in_days (Float64/float) + is_active_due (bool)  [recommended]
      - reps, lapses, factor

    Returns columns:
      objective,
      total_count, new_count, young_count, mature_count, learning_count, suspended_count, buried_count,
      next_due_days, due_today_count, overdue_count, due_7d_count, near_term_due_pressure,
      reps_sum, lapses_sum, lapse_rate, avg_factor,
      load_score
    """

    df = cards_df.copy()

    # ---- normalize tags -> list[str] ----
    def _norm_tags(ts):
        if ts is None:
            return []
        if isinstance(ts, str):
            return ts.split()
        return list(ts)

    if "tags" not in df.columns:
        raise ValueError("cards_df must include a 'tags' column.")
    df["tags_norm"] = df["tags"].apply(_norm_tags)

    # print('extractObjectiveStatistics::df')
    # print(df)

    # ---- extract objectives from tags ----
    def _extract_objectives(tag_list):
        out = []
        for t in tag_list:
            if t.startswith(objective_prefix):
                obj = t[len(objective_prefix):].strip()
                if obj:
                    out.append(obj)
        return out

    df["objectives"] = df["tags_norm"].apply(_extract_objectives)

    # Early exit if nothing matches
    if df["objectives"].apply(len).sum() == 0:
        return pd.DataFrame(columns=[
            "objective",
            "total_count", "new_count", "young_count", "mature_count",
            "learning_count", "suspended_count", "buried_count",
            "next_due_days", "due_today_count", "overdue_count", "due_7d_count", "near_term_due_pressure",
            "reps_sum", "lapses_sum", "lapse_rate", "avg_factor",
            "load_score"
        ])

    # ---- validate required scheduling fields ----
    for col in ["queue", "ivl"]:
        if col not in df.columns:
            raise ValueError(f"cards_df must include '{col}'.")

    df["queue"] = df["queue"].fillna(0).astype(int)
    df["ivl"] = df["ivl"].fillna(0).astype(int)

    # Anki queue reference (common):
    #  0=new, 1=learning, 2=review, 3=day learn, -1=suspended, -2=buried
    df["is_new"]       = (df["queue"] == 0)
    df["is_review"]    = (df["queue"] == 2)
    df["is_learning"]  = df["queue"].isin([1, 3])
    df["is_suspended"] = (df["queue"] == -1)
    df["is_buried"]    = (df["queue"] == -2)

    df["is_mature"] = df["is_review"] & (df["ivl"] >= mature_ivl_days)
    df["is_young"]  = (df["is_review"] & (df["ivl"] < mature_ivl_days)) | df["is_learning"]

    # ---- id column ----
    id_col = "card_id" if "card_id" in df.columns else None
    if id_col is None:
        df = df.reset_index().rename(columns={"index": "_row_id"})
        id_col = "_row_id"

    # ---- bring in due fields if present; else create safe empties ----
    if "is_active_due" not in df.columns:
        df["is_active_due"] = df["queue"].isin([1, 2, 3])
    if "due_in_days" not in df.columns:
        df["due_in_days"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    else:
        # Ensure nullable float so pd.NA is allowed
        try:
            df["due_in_days"] = df["due_in_days"].astype("Float64")
        except Exception:
            # fallback: coerce, turning weird values into NaN
            df["due_in_days"] = pd.to_numeric(df["due_in_days"], errors="coerce").astype("Float64")

    # ---- optional quality fields ----
    for col in ["reps", "lapses", "factor"]:
        if col not in df.columns:
            df[col] = 0

    df["reps"] = df["reps"].fillna(0).astype(int)
    df["lapses"] = df["lapses"].fillna(0).astype(int)
    # factor can be 0/None; keep numeric
    df["factor"] = pd.to_numeric(df["factor"], errors="coerce").fillna(0)

    # ---- explode objectives: one row per (card, objective) ----
    tmp = df[[
        id_col, "objectives",
        "is_new", "is_young", "is_mature", "is_learning", "is_suspended", "is_buried",
        "is_active_due", "due_in_days",
        "reps", "lapses", "factor",
    ]].copy()

    tmp = tmp.explode("objectives", ignore_index=True).rename(columns={"objectives": "objective"})
    tmp = tmp[tmp["objective"].notna() & (tmp["objective"].astype(str).str.len() > 0)]

    # Only consider due timing for active (learning/review) cards
    tmp["due_in_days_active"] = tmp["due_in_days"].where(tmp["is_active_due"] == True)

    # ---- group + aggregate ----
    out = (
        tmp.groupby("objective", as_index=False)
           .agg(
               total_count=(id_col, "count"),
               new_count=("is_new", "sum"),
               young_count=("is_young", "sum"),
               mature_count=("is_mature", "sum"),
               learning_count=("is_learning", "sum"),
               suspended_count=("is_suspended", "sum"),
               buried_count=("is_buried", "sum"),

               next_due_days=("due_in_days_active", "min"),
               due_today_count=("due_in_days_active", lambda s: (s.notna() & (s <= 0) & (s > -1)).sum()),
               overdue_count=("due_in_days_active", lambda s: (s.notna() & (s < 0)).sum()),
               due_7d_count=("due_in_days_active", lambda s: (s.notna() & (s >= 0) & (s <= 7)).sum()),

               reps_sum=("reps", "sum"),
               lapses_sum=("lapses", "sum"),
               avg_factor=("factor", "mean"),
           )
    )

    out["near_term_due_pressure"] = out["due_today_count"] + out["due_7d_count"]

    out["lapse_rate"] = (
        out["lapses_sum"] / out["reps_sum"].replace(0, pd.NA)
    ).fillna(0.0)

    out["load_score"] = (
        out["total_count"]
        + 3.0 * out["new_count"]
        + 2.0 * out["young_count"]
        + 0.5 * out["mature_count"]
    )

    out["explore_score"] = (
        2.0 * out["total_count"]
        + 1.5 * out["near_term_due_pressure"]
        + 4.0 * out["overdue_count"]
        - 3.0 * out["lapse_rate"]
        - 0.25 * out["learning_count"]
    )

    out["reinforce_score"] = (
        1.0 * out["total_count"]
        + 1.0 * out["near_term_due_pressure"]
        + 3.0 * out["overdue_count"]
        - 6.0 * out["lapse_rate"]
        - 0.75 * out["young_count"]
    )

    out = out.sort_values(
        ["near_term_due_pressure", "load_score", "objective"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return out

from wordfreq import top_n_list, zipf_frequency

KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV"}  # tweak

def build_universe(nlp, *, top_k=50_000, min_zipf=4.0):
    """
    Returns a list of dict rows:
      {lemma, pos, zipf, rank_source_word}
    Universe keys are (lemma, pos).
    """
    universe = {}
    for w in top_n_list("es", top_k):
        z = zipf_frequency(w, "es")
        if z < min_zipf:
            continue

        doc = nlp(w)
        if not doc:
            continue
        t = doc[0]
        pos = "VERB" if t.pos_ == "AUX" else t.pos_
        if pos not in KEEP_POS:
            continue

        lemma = t.lemma_.lower()
        if zipf_frequency(lemma, "es") < 1.0 and zipf_frequency(w, "es") >= 5.0:
            # suspicious lemma for a common word; keep surface as fallback or skip
            continue

        key = (lemma, pos)

        # Keep the max zipf we saw for this (lemma,pos)
        prev = universe.get(key)
        if (prev is None) or (z > prev["zipf"]):
            universe[key] = {"lemma": lemma, "pos": pos, "zipf": z, "example_surface": w}

    return list(universe.values())

from collections import defaultdict

def build_known_set_from_retrievability(card_key_map, fsrs_df, threshold=0.9, agg="max"):
    """
    card_key_map: dict card_id -> (lemma, pos)
    fsrs_df: DataFrame with columns ['card_id', 'retrievability']
    """
    buckets = defaultdict(list)

    print(fsrs_df.columns)

    for row in fsrs_df.itertuples(index=False):
        cid = row.cid
        r = row.retrievability_now
        key = card_key_map.get(cid)
        if key is None:
            continue
        buckets[key].append(r)

    known = set()
    for key, rs in buckets.items():
        if not rs:
            continue
        if agg == "max":
            score = max(rs)
        elif agg == "mean":
            score = sum(rs) / len(rs)
        elif agg == "min":
            score = min(rs)
        else:
            raise ValueError("agg must be one of: max, mean, min")

        if score >= threshold:
            known.add(key)

    return known

def zipf_band(z: float) -> str:
    if z >= 6.0: return "6.0+"
    if z >= 5.0: return "5.0–5.99"
    if z >= 4.5: return "4.5–4.99"
    if z >= 4.0: return "4.0–4.49"
    return "<4.0"

def _norm_pos(p: str) -> str:
    return str(p).strip().upper()

def _norm_lemma(s: str) -> str:
    # Keep accents (for display / identity) but normalize unicode form + whitespace
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return s

def _fold_accents(s: str) -> str:
    # Accent-insensitive matching key
    s = _norm_lemma(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def print_zipf_rows(rows, *, min_total: int = 1):
    """
    rows: list of (bucket_str, total:int, known:int, pct:float) sorted desc
    """
    for bucket, total, known, pct in rows:
        if total < min_total:
            continue
        print(f"{bucket:>10}  total={total:5d}  known={known:5d}  coverage={pct:6.2f}%")


# from collections import Counter, defaultdict
# from typing import Any, Dict, Iterable, Optional, Set, Tuple

def coverage_report(
    universe_rows: Iterable[Dict[str, Any]],
    all_cards_lemma_pos_tuples,
    known_set: Set[Tuple[str, str]],
    *,
    auto_known_zipf: Optional[float] = None,   # kept for compatibility; not used unless you want it
    pos_filter: Optional[Set[str]] = None,
    decimals: int = 2,
    accent_insensitive: bool = True,
):
    """
    Universe vs Deck vs Mastery report.

    Outputs:
      - total counts: universe / all cards you have / mastered cards
      - zipf split (>= 4.0 and < 4.0), each split into:
          * no_card
          * has_card_not_mastered
          * mastered
        (these sum to the universe count in that zipf band)
      - per-zipf-bucket rows (0.01 by default), descending:
          bucket, universe_count, have_card_count, mastered_count
    """

    pos_filter_norm = {_norm_pos(p) for p in pos_filter} if pos_filter else None

    def canon_key(lemma: str, pos: str) -> Tuple[str, str]:
        lemma_n = _norm_lemma(lemma)
        pos_n = _norm_pos(pos)
        if accent_insensitive:
            lemma_n = _fold_accents(lemma_n)
        return (lemma_n, pos_n)

    fmt = f"{{:.{decimals}f}}"

    # --- canonical sets for your deck ---
    all_cards_can = {canon_key(*card_key_map[cid]) for cid in all_cards_lemma_pos_tuples if cid in card_key_map}
    mastered_can: Set[Tuple[str, str]] = {canon_key(l, p) for (l, p) in known_set}

    # --- universe canonical + zipf per key ---
    universe_can: Set[Tuple[str, str]] = set()
    zipf_by_key: Dict[Tuple[str, str], float] = {}

    for r in universe_rows:
        lemma = r["lemma"]
        pos = r["pos"]
        z = float(r["zipf"])

        pos_n = _norm_pos(pos)
        if pos_filter_norm is not None and pos_n not in pos_filter_norm:
            continue

        key = canon_key(lemma, pos_n)
        universe_can.add(key)

        prev = zipf_by_key.get(key)
        if prev is None or z > prev:
            zipf_by_key[key] = z
    # del lemma
    # del pos
    # del z

    # --- optional "auto known" credit (off by default) ---
    # If you ever want "mastered" to include auto_known_zipf, you can enable this:
    effective_mastered_can = set(mastered_can)
    credit_only_can = set()
    # print(f"effective_mastered_can before auto known:{len(effective_mastered_can)}")
    if auto_known_zipf is not None:
        for k in universe_can:
            if zipf_by_key.get(k, float("-inf")) >= auto_known_zipf:
                effective_mastered_can.add(k)
                credit_only_can.add(k)
    else:
        effective_mastered_can = mastered_can
    # print(f"effective_mastered_can after auto known #1:{len(effective_mastered_can)}")
    # effective_mastered_can = mastered_can  # simplest: exactly what you said you wanted
    # print(f"effective_mastered_can after auto known #2:{len(effective_mastered_can)}")

    # --- totals you asked for ---
    summary = {
        "universe_count": len(universe_can),
        # both are useful: "how many cards you have" vs "how many unique lemma+pos you have"
        "all_cards_count_raw": len(list(all_cards_lemma_pos_tuples)),
        "all_cards_count_unique": len(all_cards_can),
        "mastered_count_unique": len(effective_mastered_can),
    }

    # --- zipf split breakdowns over the UNIVERSE ---
    def split_breakdown(predicate):
        no_card = 0
        has_card_not_mastered = 0

        mastered_with_card = 0
        mastered_credit_only = 0

        total = 0

        for k in universe_can:
            z = zipf_by_key.get(k, float("-inf"))
            if not predicate(z):
                continue
            total += 1

            in_deck = (k in all_cards_can)
            is_mastered = (k in effective_mastered_can)

            if is_mastered:
                if in_deck:
                    mastered_with_card += 1
                else:
                    mastered_credit_only += 1
            else:
                if in_deck:
                    has_card_not_mastered += 1
                else:
                    no_card += 1

        mastered_total = mastered_with_card + mastered_credit_only

        # invariants:
        #   no_card + has_card_not_mastered + mastered_total == total
        #   mastered_total == mastered_with_card + mastered_credit_only
        return {
            "universe_total": total,
            "no_card": no_card,
            "has_card_not_mastered": has_card_not_mastered,

            # keep a simple "mastered" for quick reading, but make it explicit
            "mastered_with_card": mastered_with_card,
            "mastered_credit_only": mastered_credit_only,
            "mastered_total": mastered_total,
        }


    high_zipf = split_breakdown(lambda z: z >= 4.0)
    low_zipf = split_breakdown(lambda z: z < 4.0)

    # --- per 0.01 zipf bucket rows (descending) ---
    by_bucket = defaultdict(lambda: {"universe": 0, "have_card": 0, "mastered": 0})

    for k in universe_can:
        z = zipf_by_key.get(k, float("-inf"))
        bucket = fmt.format(z)
        by_bucket[bucket]["universe"] += 1
        if k in all_cards_can:
            by_bucket[bucket]["have_card"] += 1
        if k in effective_mastered_can:
            by_bucket[bucket]["mastered"] += 1

    rows = []
    for bucket, d in by_bucket.items():
        rows.append((
            bucket,
            int(d["universe"]),
            int(d["have_card"]),
            int(d["mastered"]),
        ))
    rows.sort(key=lambda x: float(x[0]), reverse=True)

    deck_only_can = all_cards_can - universe_can

    not_in_zipf = {
        "universe_total": len(deck_only_can),
        "no_card": 0,
        "has_card_not_mastered": len(deck_only_can - effective_mastered_can),

        "mastered_with_card": len(deck_only_can & effective_mastered_can),
        "mastered_credit_only": 0,
        "mastered_total": len(deck_only_can & effective_mastered_can),
    }


    return {
        "summary": summary,

        "zipf_split": {
            "high_zipf_ge_4": high_zipf,
            "low_zipf_lt_4": low_zipf,
            "not_in_zipf": not_in_zipf, 
        },

        # (bucket, universe_count, have_card_count, mastered_count)
        "rows_by_zipf_bucket": rows,
    }


def top_unknown(universe_rows, known_set, *, band_min_zipf=4.5, limit=200):
    rows = [r for r in universe_rows
            if r["zipf"] >= band_min_zipf and (r["lemma"], r["pos"]) not in known_set]
    rows.sort(key=lambda r: r["zipf"], reverse=True)
    return rows[:limit]


if __name__ == "__main__":

    print('Start.')

    action = 'generate cards'
    # action = 'measure fluency'
    # action = 'inspect revlog'
    # action = 'convert 2 sided cards to 1 sided cards'
    
    DB_PATH = "/tmp/collection_ro.anki2"

    DRY_RUN = True
    SHOW_CARDS = False

    # cp "/Users/hume/Library/Application Support/Anki2/Hume/collection.anki2" /tmp/collection_ro.anki2
    if action == 'action1':
        
        

        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        decks = load_decks(cur)
        fieldmap = load_fieldmap_from_fields_table(cur)
        col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

        rows = cur.execute("""
            SELECT
                c.id AS card_id,
                c.nid,
                c.did,
                c.queue,
                c.due,
                c.ivl,
                n.flds,
                n.tags,
                n.mid
            FROM cards c
            JOIN notes n ON n.id = c.nid
            ORDER BY c.did, c.id
        """).fetchall()

        row_count = 0
        front_set = set()
        deck_dict = {}
        single_word_card_counter = 0
        for r in rows:
            row_count += 1
            mid = int(r["mid"])
            fieldnames = fieldmap.get(mid, [])
            front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

            out = {
                "deck": decks.get(int(r["did"]), f"Unknown(did={r['did']})"),
                "note_id": int(r["nid"]),
                "front": front,
                "back": back,
                "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
                "due": due_display(int(r["queue"]), int(r["due"]), col_crt),
                "tags": (r["tags"] or "").strip().split(),
            }

            if not out["deck"].startswith("Espanol::Active Learning"):
                continue

            front_set.add(out["front"])
            if out["deck"] in deck_dict.keys():
                deck_dict[out["deck"]] += 1
            else:
                deck_dict[out["deck"]] = 1
                # print(out)

            if not " " in out["front"].strip():
                pass
                if len(out["tags"]) != 0:
                    continue
                # single_word_card_counter += 1
                # print(out["front"], len(out["tags"]))
                pos_tag = chat(out["front"]) 
                print(out["front"], pos_tag)
                append_tag(out["note_id"], pos_tag)
                time.sleep(1)


            # print(out)
        # print(row_count)
        # print(len(front_set))
        # print(single_word_card_counter)
        # for k, v in deck_dict.items():
        #     print(k, v)

        con.close()

        # print(chat("Generate a spanish sentence that uses además to trigger the subjunctive."))
    elif action == 'omw test':
        ensure_wordnet_downloaded()

        pipeline = SpanishLemmaWordnet()

        text = "Los perros corren rápido, pero el perro viejo duerme."
        rows = pipeline.analyze_text(text)

        for r in rows:
            print(f"\n{r['surface']:<12} lemma={r['lemma']:<12} pos={r['pos']}")
            for h in r["wn_hits"]:
                print(f"  - {h.synset} ({h.pos}) :: {h.definition_en}")
                if h.spanish_lemmas:
                    print(f"    spa lemmas: {', '.join(h.spanish_lemmas[:8])}")  
    elif action == 'read anki cards to DF':
        pass

        # ensure_wordnet_downloaded()
        pipeline = SpanishLemmaWordnet()

        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        decks = load_decks(cur)
        fieldmap = load_fieldmap_from_fields_table(cur)
        col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

        rows = cur.execute("""
            SELECT
                c.id AS card_id,
                c.nid,
                c.did,
                c.queue,
                c.due,
                c.ivl,
                n.flds,
                n.tags,
                n.mid
            FROM cards c
            JOIN notes n ON n.id = c.nid
            ORDER BY c.did, c.id
        """).fetchall()

        row_count = 0
        card_data = []
        for r in rows:
            row_count += 1
            mid = int(r["mid"])
            fieldnames = fieldmap.get(mid, [])
            front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

            out = {
                "deck": decks.get(int(r["did"]), f"Unknown(did={r['did']})"),
                "note_id": int(r["nid"]),
                "front": front,
                "back": back,
                "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
                "due": due_display(int(r["queue"]), int(r["due"]), col_crt),
                "tags": (r["tags"] or "").strip().split(),
            }

            if not out["deck"].startswith("Espanol::Active Learning"):
                continue

            card_data.append(out)

            # if " " in out["front"].strip(): #only label cards with a single word on the front
            #     continue

            # text = out["front"]
            # rows = pipeline.analyze_text(text)

            # for r in rows:
            #     print(f"\n{r['surface']:<12} lemma={r['lemma']:<12} pos={r['pos']}")
            #     for h in r["wn_hits"]:
            #         print(f"  - {h.synset} ({h.pos}) :: {h.definition_en}")
            #         if h.spanish_lemmas:
            #             print(f"    spa lemmas: {', '.join(h.spanish_lemmas[:8])}")

            # append_tag(out["note_id"], pos_tag)

            if row_count > 10:
                break
        card_data_df = pd.DataFrame(card_data)
        print(card_data_df)

        con.close()        
    elif action == 'label anki cards EN or ES':
        pass

        pipeline = SpanishLemmaWordnet()

        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        decks = load_decks(cur)
        fieldmap = load_fieldmap_from_fields_table(cur)
        col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

        rows = cur.execute("""
            SELECT
                c.id AS card_id,
                c.nid,
                c.did,
                c.queue,
                c.due,
                c.ivl,
                n.flds,
                n.tags,
                n.mid
            FROM cards c
            JOIN notes n ON n.id = c.nid
            ORDER BY c.did, c.id
        """).fetchall()

        row_count = 0
        for r in rows:
            row_count += 1
            mid = int(r["mid"])
            fieldnames = fieldmap.get(mid, [])
            front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

            out = {
                "deck": decks.get(int(r["did"]), f"Unknown(did={r['did']})"),
                "note_id": int(r["nid"]),
                "front": front,
                "back": back,
                "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
                "due": due_display(int(r["queue"]), int(r["due"]), col_crt),
                "tags": (r["tags"] or "").strip().split(),
            }

            if not out["deck"].startswith("Espanol::Active Learning"):
                continue

            if " " in out["front"].strip(): #only label cards with a single word on the front
                continue

            text = out["front"]
            rows = pipeline.analyze_text(text)

            for r in rows:
                pass

                # print(out['tags'])
                if 'langid:es' in out['tags']:
                    continue #i already like these tags

                # spanish only caught 20% of actual, english was just not good at all
                language_guess = guess_lang_token(r['lemma'])
                print(r['lemma'], language_guess)
                if language_guess == 'es':
                    append_tag(out["note_id"], "langid:es")

                elif language_guess == 'en':
                    append_tag(out["note_id"], "langid:en")


            # append_tag(out["note_id"], pos_tag)

            # if row_count > 10:
            #     break

        con.close()
    elif action == 'label anki cards POS':
        pass

        pipeline = SpanishLemmaWordnet()

        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        decks = load_decks(cur)
        fieldmap = load_fieldmap_from_fields_table(cur)
        col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

        rows_cursor = cur.execute("""
            SELECT
                c.id AS card_id,
                c.nid,
                c.did,
                c.queue,
                c.due,
                c.ivl,
                n.flds,
                n.tags,
                n.mid
            FROM cards c
            JOIN notes n ON n.id = c.nid
            ORDER BY c.did, c.id
        """)

        row_count = 0
        assigned_pos_counts = {}
        assigned_pos_counts['UNK'] = 0
        for r in rows_cursor.fetchall():
            
            mid = int(r["mid"])
            fieldnames = fieldmap.get(mid, [])
            front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

            out = {
                "deck": decks.get(int(r["did"]), f"Unknown(did={r['did']})"),
                "note_id": int(r["nid"]),
                "front": front,
                "back": back,
                "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
                "due": due_display(int(r["queue"]), int(r["due"]), col_crt),
                "tags": (r["tags"] or "").strip().split(),
            }

            if not out["deck"].startswith("Espanol::Active Learning"):
                continue

            if " " in out["front"].strip(): #only label cards with a single word on the front
                continue

            if not 'langid:es' in dict(r)['tags'].strip().split(' '):
                continue

            text = out["front"]
            rows = pipeline.analyze_text(text)

            # print(dict(r))
            for s in rows:
                pass

                if len(s['wn_hits']) == 0:
                    assigned_pos_counts['UNK'] += 1
                    continue
                
                append_tag(out["note_id"], "posid:"+s['pos'])

                if s['pos'] in assigned_pos_counts.keys():
                    assigned_pos_counts[s['pos']] += 1
                else:
                    assigned_pos_counts[s['pos']] = 1
                break #not sure if i need this

            # row_count += 1
            # if row_count > 10:
            #     break

        for k, v in assigned_pos_counts.items():
            print(k, v)
        con.close()
    elif action == 'tag counts':
        pass

        # ensure_wordnet_downloaded()
        pipeline = SpanishLemmaWordnet()

        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        decks = load_decks(cur)
        fieldmap = load_fieldmap_from_fields_table(cur)
        col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

        rows = cur.execute("""
            SELECT
                c.id AS card_id,
                c.nid,
                c.did,
                c.queue,
                c.due,
                c.ivl,
                n.flds,
                n.tags,
                n.mid
            FROM cards c
            JOIN notes n ON n.id = c.nid
            ORDER BY c.did, c.id
        """).fetchall()

        row_count = 0
        card_data = []
        tag_hist = {}
        for r in rows:
            row_count += 1
            mid = int(r["mid"])
            fieldnames = fieldmap.get(mid, [])
            front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

            out = {
                "deck": decks.get(int(r["did"]), f"Unknown(did={r['did']})"),
                "note_id": int(r["nid"]),
                "front": front,
                "back": back,
                "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
                "due": due_display(int(r["queue"]), int(r["due"]), col_crt),
                "tags": (r["tags"] or "").strip().split(),
            }

            if not out["deck"].startswith("Espanol::Active Learning"):
                continue

            card_data.append(out)

            # if " " in out["front"].strip(): #only label cards with a single word on the front
            #     continue

            # text = out["front"]
            # rows = pipeline.analyze_text(text)

            # for r in rows:
            #     print(f"\n{r['surface']:<12} lemma={r['lemma']:<12} pos={r['pos']}")
            #     for h in r["wn_hits"]:
            #         print(f"  - {h.synset} ({h.pos}) :: {h.definition_en}")
            #         if h.spanish_lemmas:
            #             print(f"    spa lemmas: {', '.join(h.spanish_lemmas[:8])}")

            # append_tag(out["note_id"], pos_tag)

            if str(out['tags']) in tag_hist.keys():
                tag_hist[str(out['tags'])] += 1
            else:
                tag_hist[str(out['tags'])] = 1

            # if row_count > 10:
            #     break
        # card_data_df = pd.DataFrame(card_data)
        # # print(card_data_df)
        for k, v in tag_hist.items():
            print(k,v)

        con.close()        
    elif action == 'read anki cards to DF':
        pass

        # ensure_wordnet_downloaded()
        # pipeline = SpanishLemmaWordnet()

        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        decks = load_decks(cur)
        fieldmap = load_fieldmap_from_fields_table(cur)
        col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

        rows = cur.execute("""
            SELECT
                c.id AS card_id,
                c.nid,
                c.did,
                c.queue,
                c.due,
                c.ivl,
                n.flds,
                n.tags,
                n.mid
            FROM cards c
            JOIN notes n ON n.id = c.nid
            ORDER BY c.did, c.id
        """).fetchall()

        row_count = 0
        card_data = []
        for r in rows:
            row_count += 1
            mid = int(r["mid"])
            fieldnames = fieldmap.get(mid, [])
            front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

            out = {
                "deck": decks.get(int(r["did"]), f"Unknown(did={r['did']})"),
                "note_id": int(r["nid"]),
                "front": front,
                "back": back,
                "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
                "due": due_display(int(r["queue"]), int(r["due"]), col_crt),
                "tags": (r["tags"] or "").strip().split(),
            }

            if not out["deck"].startswith("Espanol::Active Learning"):
                continue

            card_data.append(out)

            # if " " in out["front"].strip(): #only label cards with a single word on the front
            #     continue

            # text = out["front"]
            # rows = pipeline.analyze_text(text)

            # for r in rows:
            #     print(f"\n{r['surface']:<12} lemma={r['lemma']:<12} pos={r['pos']}")
            #     for h in r["wn_hits"]:
            #         print(f"  - {h.synset} ({h.pos}) :: {h.definition_en}")
            #         if h.spanish_lemmas:
            #             print(f"    spa lemmas: {', '.join(h.spanish_lemmas[:8])}")

            # append_tag(out["note_id"], pos_tag)

            if row_count > 10:
                break
    elif action == 'generate cards':
        pass

        ### this was copy pasted from another project that's why it's ugly
        irregular_ar_verbs__dict = {}
        irregular_er_verbs__dict = {}
        irregular_ir_verbs__dict = {}

        irregular_ar_verbs__dict['estar'] = 'to be (temporary)'
        irregular_ar_verbs__dict['andar'] = 'to walk'
        irregular_ar_verbs__dict['dar'] = 'to give'
        irregular_ar_verbs__dict['jugar'] = 'to play'
        irregular_ar_verbs__dict['regar'] = 'to water'
        irregular_ar_verbs__dict['negar'] = 'to deny'
        irregular_ar_verbs__dict['empezar'] = 'to start'
        irregular_ar_verbs__dict['calentar'] = 'to heat'
        irregular_ar_verbs__dict['aferrar'] = 'to grasp'
        irregular_ar_verbs__dict['pensar'] = 'to think'
        irregular_ar_verbs__dict['recalentar'] = 'to warm up'
        irregular_ar_verbs__dict['recomendar'] = 'to recommend'
        irregular_ar_verbs__dict['reventar'] = 'to burst'
        irregular_ar_verbs__dict['sentar'] = 'to sit'
        irregular_ar_verbs__dict['tentar'] = 'to tempt'
        irregular_ar_verbs__dict['acertar'] = 'to get right; to guess correctly'
        irregular_ar_verbs__dict['soterrar'] = 'to bury'
        irregular_ar_verbs__dict['manifestar'] = 'to declare'
        irregular_ar_verbs__dict['quebrar'] = 'to break; to go bankrupt'
        irregular_ar_verbs__dict['sembrar'] = 'to plant; to sow'
        irregular_ar_verbs__dict['temblar'] = 'to shake'
        irregular_ar_verbs__dict['comenzar'] = 'to begin; to start'
        irregular_ar_verbs__dict['acostar'] = 'to put to bed'
        irregular_ar_verbs__dict['colgar'] = 'to hang'
        irregular_ar_verbs__dict['renovar'] = 'to renew'
        irregular_ar_verbs__dict['apostar'] = 'to bet'
        irregular_ar_verbs__dict['contar'] = 'to count; to tell'
        irregular_ar_verbs__dict['mostrar'] = 'to show'
        irregular_ar_verbs__dict['poblar'] = 'to populate; to fill'
        irregular_ar_verbs__dict['probar'] = 'to try; to taste'
        irregular_ar_verbs__dict['recordar'] = 'to remember'
        irregular_ar_verbs__dict['rodar'] = 'to roll; to shoot (as in film)'
        irregular_ar_verbs__dict['soldar'] = 'to solder'
        irregular_ar_verbs__dict['soltar'] = 'to let go of; to loosen'
        irregular_ar_verbs__dict['sonar'] = 'to ring'
        irregular_ar_verbs__dict['soñar'] = 'to dream'
        irregular_ar_verbs__dict['tostar'] = 'to toast'
        irregular_ar_verbs__dict['volar'] = 'to fly'
        irregular_ar_verbs__dict['almorzar'] = 'to have lunch'
        irregular_ar_verbs__dict['sosegar'] = 'to calm'
        irregular_ar_verbs__dict['reforzar'] = 'to reinforce'

        irregular_er_verbs__dict['caber'] = 'to fit'
        irregular_er_verbs__dict['haber'] = 'to be obligated to; (used in compound tenses)'
        irregular_er_verbs__dict['oler'] = 'to smell'
        irregular_er_verbs__dict['saber'] = 'to know; to taste'
        irregular_er_verbs__dict['ser'] = 'to be (permanent)'
        irregular_er_verbs__dict['valer'] = 'to be worth; to cost'
        irregular_er_verbs__dict['ver'] = 'to see; to watch'
        irregular_er_verbs__dict['deshacer'] = 'to undo'
        irregular_er_verbs__dict['hacer'] = 'to do; to make'
        irregular_er_verbs__dict['satisfacer'] = 'to satisfy'
        irregular_er_verbs__dict['contener'] = 'to contain'
        irregular_er_verbs__dict['detener'] = 'to stop'
        irregular_er_verbs__dict['disponer'] = 'to arrange; to have'
        irregular_er_verbs__dict['entretener'] = 'to entertain'
        irregular_er_verbs__dict['exponer'] = 'to expose; to exhibit'
        irregular_er_verbs__dict['imponer'] = 'to impose'
        irregular_er_verbs__dict['mantener'] = 'to hold'
        irregular_er_verbs__dict['oponer'] = 'to put up'
        irregular_er_verbs__dict['poner'] = 'to put'
        irregular_er_verbs__dict['posponer'] = 'to postpone'
        irregular_er_verbs__dict['predisponer'] = 'to predispose'
        irregular_er_verbs__dict['proponer'] = 'to propose'
        irregular_er_verbs__dict['reponer'] = 'to replace'
        irregular_er_verbs__dict['retener'] = 'to keep; to retain'
        irregular_er_verbs__dict['sobreponer'] = 'to put on top of'
        irregular_er_verbs__dict['sostener'] = 'to hold'
        irregular_er_verbs__dict['suponer'] = 'to suppose'
        irregular_er_verbs__dict['tener'] = 'to have; to be'
        #irregular_er_verbs__dict['atener'] = ''
        irregular_er_verbs__dict['atraer'] = 'to attract'
        irregular_er_verbs__dict['caer'] = 'to fall'
        irregular_er_verbs__dict['distraer'] = 'to distract'
        irregular_er_verbs__dict['extraer'] = 'to extract; to draw'
        irregular_er_verbs__dict['recaer'] = 'to suffer a relapse'
        irregular_er_verbs__dict['retraer'] = 'to retract'
        irregular_er_verbs__dict['sustraer'] = 'to subtract'
        irregular_er_verbs__dict['traer'] = 'to bring'
        irregular_er_verbs__dict['contraer'] = 'to contract'
        irregular_er_verbs__dict['poseer'] = 'to have'
        irregular_er_verbs__dict['proveer'] = 'to provide'
        irregular_er_verbs__dict['leer'] = 'to read'
        irregular_er_verbs__dict['creer'] = 'to believe'
        irregular_er_verbs__dict['conmover'] = 'to move'
        irregular_er_verbs__dict['devolver'] = 'to give back; to return'
        irregular_er_verbs__dict['volver'] = 'to return'
        irregular_er_verbs__dict['desvolver'] = 'to plow'
        irregular_er_verbs__dict['envolver'] = 'to wrap'
        irregular_er_verbs__dict['mover'] = 'to move'
        irregular_er_verbs__dict['promover'] = 'to promote'
        irregular_er_verbs__dict['resolver'] = 'to solve'
        irregular_er_verbs__dict['poder'] = 'to be able to'
        irregular_er_verbs__dict['morder'] = 'to bite'
        irregular_er_verbs__dict['llover'] = 'to rain'
        irregular_er_verbs__dict['doler'] = 'to hurt'
        irregular_er_verbs__dict['demoler'] = 'to demolish'
        irregular_er_verbs__dict['verter'] = 'to pour; to spill'
        irregular_er_verbs__dict['tender'] = 'to hang'
        irregular_er_verbs__dict['querer'] = 'to want'
        irregular_er_verbs__dict['malentender'] = 'to misunderstand'
        irregular_er_verbs__dict['extender'] = 'to spread out'
        irregular_er_verbs__dict['entender'] = 'to understand'
        irregular_er_verbs__dict['desatender'] = 'to neglect; to disregard'
        irregular_er_verbs__dict['defender'] = 'to defend'
        irregular_er_verbs__dict['contender'] = 'to contend'

        irregular_ir_verbs__dict['ir'] = 'to go'
        irregular_ir_verbs__dict['adherir'] = 'to stick'
        irregular_ir_verbs__dict['advertir'] = 'to warn'
        irregular_ir_verbs__dict['adquirir'] = 'to acquire; to purchase'
        irregular_ir_verbs__dict['diferir'] = 'to differ; to postpone'
        irregular_ir_verbs__dict['digerir'] = 'to digest; to assimilate'
        irregular_ir_verbs__dict['herir'] = 'to wound; to hurt'
        irregular_ir_verbs__dict['inferir'] = 'to infer'
        irregular_ir_verbs__dict['ingerir'] = 'to ingest'
        irregular_ir_verbs__dict['interferir'] = 'to interfere with'
        irregular_ir_verbs__dict['preferir'] = 'to prefer'
        irregular_ir_verbs__dict['referir'] = 'to refer'
        irregular_ir_verbs__dict['transferir'] = 'to transfer'
        irregular_ir_verbs__dict['discernir'] = 'to discern; to distinguish'
        irregular_ir_verbs__dict['divertir'] = 'to amuse; to entertain'
        irregular_ir_verbs__dict['sentir'] = 'to feel'
        irregular_ir_verbs__dict['revertir'] = 'to revert to'
        irregular_ir_verbs__dict['resentir'] = 'to resent'
        irregular_ir_verbs__dict['presentir'] = 'to have a feeling; to sense'
        irregular_ir_verbs__dict['mentir'] = 'to lie'
        irregular_ir_verbs__dict['invertir'] = 'to invest'
        irregular_ir_verbs__dict['rugir'] = 'to roar'
        irregular_ir_verbs__dict['sumergir'] = 'to submerge'
        irregular_ir_verbs__dict['surgir'] = 'to arise'
        irregular_ir_verbs__dict['transigir'] = 'to compromise'
        irregular_ir_verbs__dict['ungir'] = 'to put ointment on; to elect'
        irregular_ir_verbs__dict['dirigir'] = 'to manage; to run'
        irregular_ir_verbs__dict['exigir'] = 'to demand; to call for'
        irregular_ir_verbs__dict['infringir'] = 'to infringe'
        irregular_ir_verbs__dict['producir'] = 'to produce'
        irregular_ir_verbs__dict['reducir'] = 'to reduce'
        irregular_ir_verbs__dict['seducir'] = 'to seduce'
        irregular_ir_verbs__dict['traducir'] = 'to translate'
        irregular_ir_verbs__dict['inducir'] = 'to lead to; to induce'
        irregular_ir_verbs__dict['introducir'] = 'to insert'
        irregular_ir_verbs__dict['lucir'] = 'to shine; to wear'
        irregular_ir_verbs__dict['perseguir'] = 'to pursue'
        irregular_ir_verbs__dict['distinguir'] = 'to distinguish; to tell the difference'
        irregular_ir_verbs__dict['seguir'] = 'to follow'
        irregular_ir_verbs__dict['ocluir'] = 'to occlude'
        irregular_ir_verbs__dict['destruir'] = 'to destroy'
        irregular_ir_verbs__dict['disminuir'] = 'to reduce; to decrease'
        irregular_ir_verbs__dict['distribuir'] = 'to distribute'
        irregular_ir_verbs__dict['excluir'] = 'to exclude'
        irregular_ir_verbs__dict['fluir'] = 'to flow'
        irregular_ir_verbs__dict['huir'] = 'to escape'
        irregular_ir_verbs__dict['incluir'] = 'to include'
        irregular_ir_verbs__dict['influir'] = 'to influence'
        irregular_ir_verbs__dict['instituir'] = 'to institute'
        irregular_ir_verbs__dict['sustituir'] = 'to substitute; to replace'
        irregular_ir_verbs__dict['teñir'] = 'to dye'
        irregular_ir_verbs__dict['elegir'] = 'to choose'
        irregular_ir_verbs__dict['regir'] = 'to govern'
        irregular_ir_verbs__dict['pedir'] = 'to ask for; to order'
        # irregular_ir_verbs__dict['dervetir'] = '' ?
        irregular_ir_verbs__dict['despedir'] = 'to say goodbye to; to fire'
        irregular_ir_verbs__dict['freír'] = 'to fry'
        irregular_ir_verbs__dict['gemir'] = 'to moan'
        irregular_ir_verbs__dict['impedir'] = 'to prevent'
        irregular_ir_verbs__dict['servir'] = 'to be useful; to serve'
        irregular_ir_verbs__dict['repetir'] = 'to do again; to repeat'
        irregular_ir_verbs__dict['rendir'] = 'to produce'
        irregular_ir_verbs__dict['reñir'] = 'to tell off'
        irregular_ir_verbs__dict['medir'] = 'to measure'
        irregular_ir_verbs__dict['vestir'] = 'to wear'
        irregular_ir_verbs__dict['decir'] = 'to say; to tell'
        irregular_ir_verbs__dict['predecir'] = 'to predict'
        irregular_ir_verbs__dict['venir'] = 'to come'
        irregular_ir_verbs__dict['oír'] = 'to hear'
        irregular_ir_verbs__dict['salir'] = 'to leave'
        irregular_ir_verbs__dict['movir'] = 'to go out; to leave'
        irregular_ir_verbs__dict['sobresalir'] = 'to stick our; to stand out'
        irregular_ir_verbs__dict['intervenir'] = 'to intervene'
        irregular_ir_verbs__dict['dormir'] = 'to sleep'

        irregular_verbs_es_en = dict(irregular_ar_verbs__dict,**irregular_er_verbs__dict,**irregular_ir_verbs__dict)
        irregular_verbs_en_es = {v: k for k, v in irregular_verbs_es_en.items()}

        cards = getAnkiCards()
        noun_usage_stats, verb_usage_stats = extractSeedStatistics(cards, mature_ivl_days=21)
        verb_usage_stats = add_hardcoded_seeds(verb_usage_stats, set(irregular_verbs_es_en.keys()) )
        # print('noun_usage_stats:')
        # print(noun_usage_stats)
        # print('verb_usage_stats:')
        # print(verb_usage_stats)
        pair_gen = make_generators(noun_usage_stats, verb_usage_stats)
        # e.g. noun, verb = next(pair_gen)

        # for _, row in cards.iterrows():
        #     tags = row.loc['tags']
        #     if not 'langid:es' in tags:
        #         continue
        #     if 'posid:NOUN' in tags:
        #         noun_set.add(row["front"])
        #     elif 'posid:VERB' in tags:
        #         verb_set.add(row["front"])
        
        deck_and_prompt_tuples = []

        ### CLOZE
        # TODO que vs. quien vs. lo que
        # TODO a vs. (no a)
        # TODO uses of se: reflexive and pronominal
        # TODO emotion vocab alone
        # TODO emotion words (in a sentence) and con vs. por vs. de
        # TODO que / quien / el que / lo que / el cual (esp. relative clauses with/without antecedent)
        # por vs. para with verbs that bias one (luchar por, optar por, servir para, estar para)
        # ser + adjective vs. estar + adjective (meaning shift) ( e.g. listo, aburrido, interesado, seguro, consciente, atento )

        # Some more ideas from gpt:
        # Verb + indirect object vs. prepositional object
        # ayudar a
        # pedirle algo a alguien
        # robarle algo a alguien
        # Le / lo / la confusion
        # esp. with people + things
        # Accidental se vs. passive se (contrast explicitly)
        # Preterite vs. imperfect with meaning change ( e.g. sabía / supe, podía / pude, quería / quise )
        # Future vs. ir a + infinitive (intention vs. prediction)
        # Conditional for politeness / conjecture
        #
        # Assertion-blocking adjectives ( e.g. es probable que / es evidente que )
        # Relative clauses with existence (e.g. busco algo que sea… )
        # Subjunctive in commands / indirect commands (e.g. que pase )
        # Past subjunctive sequence of tense ( e.g. quería que viniera )

        # Unintuitive polysemy
        #

        # Periphrastic verbs
        # acabar de
        # llevar + gerund
        # venir + gerund

        # Idioms that encode grammar
        # tener ganas de
        # darse cuenta de
        # hacer falta

        
        # og_noun_set_size = len(noun_set)
        # og_verb_set_size = len(verb_set)

        ser_use_cases = [
            'Identity',
            'Definition',
            'Essential Characteristics',
            'Origin',
            'Material (what is made of)',
            'Possession',
            'Relationship',
            'Time, Date, Events',
            'Passive Voice',
            'Impersonal / Evaluative Statements'
        ]
        estar_use_cases = [
            'Physical or emotional state',
            # 'Location of People and Objects',
            'Progressive Aspect',
            'Result of an Action',
            'Temporary / Contextual Manifestations'
        ]

        for ser_use_case in ser_use_cases:
            noun, verb = next(pair_gen)
            words_list = noun+', '+verb

            ser_prompt = f"Generate a sentence that uses \"ser\" because of {ser_use_case} and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Ser - {ser_use_case}" , (ser_prompt, words_list)))
            del noun
            del verb
            del words_list
        del ser_use_case

        for estar_use_case in estar_use_cases:
            noun, verb = next(pair_gen)
            words_list = noun+', '+verb

            estar_prompt = f"Generate a sentence that uses \"estar\" because of {estar_use_case} and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Estar - {estar_use_case}" , (estar_prompt, words_list)))
            del noun
            del verb
            del words_list
        del estar_use_case


        se_triggers = [#'Reflexive se','Pronominal', # i feel like this should be handled separately
                       'Reciprocal se', 'Accidental / Unintended se', 'Passive se', 'Impersonal se']
        
        for se_trigger in se_triggers:
            noun, verb = next(pair_gen)
            words_list = noun+', '+verb

            se_prompt_preamble_1 = f"Generate a sentence that uses \"se\" because of {se_trigger} and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Se - {se_trigger}" , (se_prompt_preamble_1, words_list)))
            del noun
            del verb
            del words_list
        del se_trigger


        # subjunctive_prompt_preamble_1 = "Generate a sentence that uses the subjunctive trigger phrase ({subj_trigger}) and contains these words or conjugations of these words ({words_list})."

        subj_triggers_1 = ["Volition / Influence","Emotion / Value Judgement","Doubt / Uncertainty",
                         "Purpose / Condition (Time specifically)","Purpose / Condition (Condition specifically)",
                         "Non-existent / indefinite antecedent"]

        for subj_trigger in subj_triggers_1:
            noun, verb = next(pair_gen)
            words_list = noun+', '+verb

            subjunctive_prompt_preamble_1 = f"Generate a sentence that uses the subjunctive because of {subj_trigger} and contains these words or conjugations of these words: {words_list}."
            #print(subjunctive_prompt_preamble_1)
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , (subjunctive_prompt_preamble_1, words_list)))
            del noun
            del verb
            del words_list

        del subj_trigger
        
        sometimes_subj_triggers = ["cuando","mientras que","hasta que","después de que","tan pronto como"]

        for subj_trigger in sometimes_subj_triggers:
            noun, verb = next(pair_gen)
            words_list = noun+', '+verb
            subjunctive_prompt_preamble_2 = f"Generate a sentence that uses the subjunctive trigger phrase \"{subj_trigger}\" because the statement is in the future or not realized and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , (subjunctive_prompt_preamble_2, words_list)))
            del noun
            del verb
            del words_list

            noun, verb = next(pair_gen)
            words_list = noun+', '+verb
            subjunctive_prompt_preamble_3 = f"Generate a sentence that uses the phrase \"{subj_trigger}\" but does not trigger the subjunctive because the statement is habitual or in the past and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , (subjunctive_prompt_preamble_3, words_list)))
            del noun
            del verb
            del words_list
            # print(subjunctive_prompt_preamble_2)
            # print(subjunctive_prompt_preamble_3)

        del subj_trigger

        depends_subj_triggers = ["aunque",
                                #  "aun así","y eso que" #GPT reviewed my code and said these were not a good choice
                                 ]

        for subj_trigger in depends_subj_triggers:
            noun, verb = next(pair_gen)
            words_list = noun+', '+verb
            subjunctive_prompt_preamble_4 = f"Generate a sentence that uses the phrase \"{subj_trigger}\" but DOES NOT use the subjunctive because of the intention of the speaker and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , (subjunctive_prompt_preamble_4, words_list)))
            del noun
            del verb
            del words_list
            
            noun, verb = next(pair_gen)
            words_list = noun+', '+verb
            subjunctive_prompt_preamble_5 = f"Generate a sentence that uses the phrase \"{subj_trigger}\" and DOES use the subjunctive because of the intention of the speaker and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , (subjunctive_prompt_preamble_5, words_list)))
            del noun
            del verb
            del words_list
            
            # print(subjunctive_prompt_preamble_4)
            # print(subjunctive_prompt_preamble_5)

        del subj_trigger

        noun, verb = next(pair_gen)
        words_list = noun+', '+verb
        subjunctive_prompt_preamble_6 = f"Generate a sentence that uses the imperfect subjunctive and contains these words or conjugations of these words: {words_list}."
        deck_and_prompt_tuples.append((f"Subj - Imperfect" , (subjunctive_prompt_preamble_6, words_list)))
        del noun
        del verb
        del words_list

        noun, verb = next(pair_gen)
        words_list = noun+', '+verb
        subjunctive_prompt_preamble_7 = f"Generate a sentence that uses the pluperfect subjunctive and contains these words or conjugations of these words: {words_list}."
        deck_and_prompt_tuples.append((f"Subj - Pluperfect" , (subjunctive_prompt_preamble_7, words_list)))
        del noun
        del verb
        del words_list

        # print(subjunctive_prompt_preamble_6)
        # print(subjunctive_prompt_preamble_7)

        complex_coordination_dict = {}
        complex_coordination_dict["Concessive / Contrastive"] = [
        'a pesar de',
        'pese a',
        'aun con',
        'incluso con',
        'con todo',
        'sin embargo (adverbial connector)',
        'no obstante',
        'aun así',
        'así y todo',
        'a pesar de que',
        'pese a que',
        'aunque',
        'aun cuando',
        'incluso cuando',
        'si bien',
        'por más que',
        'por mucho que',
        'mal que',
        'bien que (literary)',
        'y eso que'
        ]
        complex_coordination_dict["Causal"] = [
            # 'porque',
            'ya que',
            'puesto que',
            'dado que',
            'debido a que',
            'a causa de que',
            'como (sentence-initial)',
            'a causa de',
            'debido a',
            'por culpa de',
            'gracias a',
            'por razón de'
        ]
        complex_coordination_dict["Final / Purpose"] = [
            'para que',
            'a fin de que',
            'con el fin de que',
            'con objeto de que',
            'para',
            'a fin de',
            'con el fin de',
            'con objeto de',
            'con vistas a'
        ]
        complex_coordination_dict["Conditional / Hypothetical"] = [
            'a menos que',
            'a no ser que',
            'con tal de que',
            'siempre que',
            'siempre y cuando',
            'en caso de que',
            'en el supuesto de que',
            'salvo que',
            'excepto que',
            'de no ser que (formal)',
            'en caso de',
            'con tal de',
            'a condición de'
        ]
        complex_coordination_dict["Temporal"] = [
            'cuando',
            # 'mientras',
            # 'mientras que', # a la GPT "mientras que is usually contrastive (“whereas”) more than temporal; mientras is the temporal one."
            'en cuanto',
            'tan pronto como',
            'apenas',
            'una vez que',
            'desde que',
            'hasta que',
            'antes de que',
            'después de que',
            'antes de',
            'después de',
            'al (al entrar, al salir)',
            'durante',
            'a lo largo de'
        ]
        complex_coordination_dict["Consequence / Result"] = [
            'así que',
            'por lo tanto',
            'por consiguiente',
            'de modo que',
            'de manera que',
            'de forma que',
            'conque',
            'entonces',
            'así pues'
        ] 
        
        complex_coordination_dict["Comparative / Proportional"] = [
            'como si',
            'igual que',
            'al igual que',
            'tal como',
            'así como',
            'más… que',
            'menos… que',
            'tan… como',
            'tanto… como',
            'cuanto más…, más…',
            'cuanto menos…, menos…'
        ]
        complex_coordination_dict["Additive / Emphatic"] = [
            'además de',
            'además',
            'encima',
            'por si fuera poco',
            'es más',
            'incluso',
            'hasta',
            'aun',
            'ni siquiera'
        ]
        complex_coordination_dict["Adversative / Correction"] = [
            'pero',
            'sino',
            'sino que',
            'más bien',
            'en cambio',
            'por el contrario',
            'antes bien'
        ]
        complex_coordination_dict["Explanation / Clarification"] = [
            'es que',
            'o sea',
            'es decir',
            'esto es',
            'mejor dicho',
            # 'dicho de otro modo' #obvious
        ]
        complex_coordination_dict["Restriction / Limitation"] = [
            'en la medida en que',
            'hasta cierto punto',
            'si acaso',
            'como mucho',
            'a lo sumo',
            'por lo menos',
            'cuando menos'
        ]
        complex_coordination_dict["Modal / Attitudinal"] = [
            'al parecer',
            'por lo visto',
            'según parece',
            'a mi parecer',
            'en mi opinión',
            'desde mi punto de vista',
            'hasta donde sé'
        ]
        complex_coordination_dict["Fixed causative & periphrastic constructions"] = [
            'hacer que',
            'dejar que',
            'permitir que',
            'impedir que',
            'evitar que',
            'lograr que',
            'conseguir que',
            'provocar que',
            'dar lugar a que',
            'llevar a que'
        ]

        for comp_coord_cat, phrase_list in complex_coordination_dict.items():
            for phrase in phrase_list:
                noun, verb = next(pair_gen)
                words_list = noun+', '+verb
                comp_coord_prompt = f"Generate a sentence that uses \"{phrase}\" in the \"{comp_coord_cat}\" sense and contains these words or conjugations of these words: {words_list}."
                deck_and_prompt_tuples.append((f"Comp Coord - {phrase}" , (comp_coord_prompt, words_list)))
                del noun
                del verb
                del words_list
                # print(comp_coord_prompt)

        # deck = {
        #     "hola": {"back": "hello", "tags": ["spanish", "greeting"]},
        #     "adiós": ("goodbye", ["spanish"]),
        #     "gracias": "thanks",  # no tags still works
        # }

        #Translate
        deck_name_to_card_dict = {}
        for d_p in deck_and_prompt_tuples:
            deck_specific_name = str(d_p[0])
            deck_full_name = 'Dave LLM Sentences :: '+str(d_p[0])
            spanish_sentence_prompt = d_p[1][0]
            words_list = d_p[1][1]
            noun = words_list.split(',')[0].strip()
            verb = words_list.split(',')[1].strip()
            
            if DRY_RUN:
                spanish_sentence = "Spanish Sentence Placeholder: "+str(spanish_sentence_prompt)
                english_sentence = "English Sentence Placeholder: "+str(spanish_sentence_prompt)
            else:
                spanish_sentence = chat(spanish_sentence_prompt)
                english_sentence = translate_es_to_en(spanish_sentence)
            
            if SHOW_CARDS:
                print(spanish_sentence)
                print(english_sentence)
                print('-----------------------------------')
                print('')
            # time.sleep(1)

            if not deck_full_name in deck_name_to_card_dict.keys():
                deck_name_to_card_dict[deck_full_name] = {}
            
            deck_name_to_card_dict[deck_full_name][english_sentence] = {"back": spanish_sentence,
                                                                       "tags":["learningobjective:"+deck_specific_name.replace(" ","_"),
                                                                               "notconfirmed",
                                                                                "nounseed:"+noun,
                                                                                "verbseed:"+verb,
                                                                               ]}
        del deck_specific_name
        del deck_full_name
        del spanish_sentence_prompt
        del spanish_sentence
        del english_sentence
        del noun
        del verb

        por_reasons = ["Purpose / Cause",
            # "Exchange",
            "Reason / Motive",
            "Frequency / Duration",
            "Emotions / Opinions",
            # "Communication / Means",
            "Travel / Through"]
        para_reasons = [
            # "Aim / Purpose",
            "Target / Destination",
            "Time limit / Deadline",
            "Recipient",
            # "Audience / Use",
            # "Comparison / Opinion",
            "Toward (movement)",
            # "Expectation / Outcome",
            "Destination / Direction"
            ]

        for por_reason in por_reasons:
            deck_specific_name = f"Por - {por_reason}"
            deck_full_name = 'Dave LLM Sentences :: '+deck_specific_name
            if not deck_full_name in deck_name_to_card_dict.keys():
                deck_name_to_card_dict[deck_full_name] = {}

            noun, verb = next(pair_gen)
            words_list = noun+', '+verb
            cloze_por_prompt = f"Generate a sentence that uses \"por\" exactly once in the sense of \"{por_reason}\" and contains these words or conjugations of these words ({words_list})."
            if DRY_RUN:
                spanish_sentence = 'Spanish Sentence Placeholder: '+str(cloze_por_prompt)
            else:
                spanish_sentence = chat(cloze_por_prompt)
                english_sentence = translate_es_to_en(spanish_sentence)
            spanish_sentence = re.sub(r'\b[Pp]or\b', '____', spanish_sentence, count=1)
            
            deck_name_to_card_dict[deck_full_name][spanish_sentence] = {"back": deck_specific_name+" ; "+spanish_sentence,
                                                                       "tags":["learningobjective:"+deck_specific_name.replace(" ","_"),
                                                                               "notconfirmed",
                                                                                "nounseed:"+noun,
                                                                                "verbseed:"+verb,
                                                                               ]}

        del por_reason
        del noun
        del verb

        for para_reason in para_reasons:
            deck_specific_name = f"Para - {para_reason}"
            deck_full_name = 'Dave LLM Sentences :: '+deck_specific_name
            if not deck_full_name in deck_name_to_card_dict.keys():
                deck_name_to_card_dict[deck_full_name] = {}

            noun, verb = next(pair_gen)
            words_list = noun+', '+verb
            cloze_para_prompt = f"Generate a sentence that uses \"para\" exactly once in the sense of \"{para_reason}\" and contains these words or conjugations of these words ({words_list})."
            
            if DRY_RUN:
                spanish_sentence = 'Spanish Sentence Placeholder: '+str(cloze_para_prompt)
            else:
                spanish_sentence = chat(cloze_para_prompt)
                english_sentence = translate_es_to_en(spanish_sentence)
            spanish_sentence = re.sub(r'\b[Pp]ara\b', '____', spanish_sentence, count=1)

            
            deck_name_to_card_dict[deck_full_name][spanish_sentence] = {"back": deck_specific_name+" ; "+spanish_sentence,
                                                                       "tags":["learningobjective:"+deck_specific_name.replace(" ","_"),
                                                                               "notconfirmed",
                                                                                "nounseed:"+noun,
                                                                                "verbseed:"+verb
                                                                                ]}
        del para_reason
        del noun
        del verb
        
        objective_stats = extractObjectiveStatistics(cards)
        # print(objective_stats.to_string())
        # print(objective_stats['objective'].to_string()) #explore_score, reinforce_score

        #TODO here we can iterate over deck_name_to_card_dict and reduce output as needed
        # I want por, para, se, ser, and estar groups chosen as a whole
        # 135               Estar_-_Location_of_People_and_Objects
        # 136                  Estar_-_Physical_or_emotional_state
        # 137                           Estar_-_Progressive_Aspect
        # 138                          Estar_-_Result_of_an_Action
        # 139        Estar_-_Temporary_/_Contextual_Manifestations
        # 140                                 Para_-_Aim_/_Purpose
        # 141                                Para_-_Audience_/_Use
        # 142                          Para_-_Comparison_/_Opinion
        # 143                       Para_-_Destination_/_Direction
        # 144                         Para_-_Expectation_/_Outcome
        # 145                                     Para_-_Recipient
        # 146                          Para_-_Target_/_Destination
        # 147                         Para_-_Time_limit_/_Deadline
        # 148                             Para_-_Toward_(movement)
        # 149                          Por_-_Communication_/_Means
        # 150                            Por_-_Emotions_/_Opinions
        # 151                                       Por_-_Exchange
        # 152                           Por_-_Frequency_/_Duration
        # 153                                Por_-_Purpose_/_Cause
        # 154                                Por_-_Reason_/_Motive
        # 155                               Por_-_Travel_/_Through
        # 156                      Se_-_Accidental_/_Unintended_se
        # 157                                   Se_-_Impersonal_se
        # 158                                      Se_-_Passive_se
        # 159                                   Se_-_Reciprocal_se
        # 160                                     Ser_-_Definition
        # 161                      Ser_-_Essential_Characteristics
        # 162                                       Ser_-_Identity
        # 163             Ser_-_Impersonal_/_Evaluative_Statements
        # 164                     Ser_-_Material_(what_is_made_of)
        # 165                                         Ser_-_Origin
        # 166                                  Ser_-_Passive_Voice
        # 167                                     Ser_-_Possession
        # 168                                   Ser_-_Relationship
        # 169                             Ser_-_Time,_Date,_Events
        
        deck_list = []
        counter = 0
        for k, v in deck_name_to_card_dict.items():
            # print(k)
            # print(counter)
            counter += 1
            deck_list.append(build_anki_deck(k,v))

        print('Num Cards Created: '+str(counter))

        ### So, generate an .apkg that is a collect of a few cards for many different decks
        # build_anki_deck(deck_name, deck_dict)

        pkg = genanki.Package(deck_list)
        if DRY_RUN:
            pkg.write_to_file('test_deck.apkg')
            print('Wrote test_deck.apkg')
        else:
            pkg.write_to_file('IRL_deck.apkg')
            print('Wrote IRL_deck.apkg')
        
        # print('Count Nouns: '+str(len(noun_usage_stats)))
        # print('Count Verbs: '+str(len(verb_usage_stats)))
    elif action == 'append auto-lemma seed tags':
        pass

        slw = SpanishLemmaWordnet()

        sentences_df = getAnkiSentenceCards()
        print(sentences_df.columns)
        print(sentences_df)

        for index, row in sentences_df.iterrows():
            pass

            continue_signal = False
            for existing_tag in row['tags']:
                if existing_tag.startswith('auto-lemma'):
                    continue_signal = True
                    break
            if continue_signal:
                continue

            spanish_sentence = row['back']
            note_id = row['note_id']

            lemma_pos_tuples = exractLemmaPOSTuplesFromSentence(spanish_sentence)
            raise NotImplementedError

            
            tuples = [
                (lemma, pos)
                for _, lemma, pos in slw.lemmatize(spanish_sentence)
                if pos in ("NOUN", "VERB")   # keep only NOUN/VERB
            ]

            for t in tuples:
                lemma = t[0]
                pos = t[1]

                if pos == 'NOUN':
                    append_tag(note_id, "auto-lemma-noun:"+str(lemma))

                elif pos == 'VERB':
                    # if the lemma of the verb doesn't end in r it's not real
                    if not lemma.endswith('r'):
                        continue
                    append_tag(note_id, "auto-lemma-verb:"+str(lemma))

            print(spanish_sentence)
            print(tuples)
            print('-----------------------------------------------------------------------')
            time.sleep(1)

            # todo 
            # e.g. append_tag(out["note_id"], "auto-lemma-verb:VERB") 
            #todo e.g. append_tag(out["note_id"], "auto-lemma-noun:NOUN") 
    elif action == 'measure fluency':
        nlp = spacy.load("es_core_news_sm")

        cards = getAnkiCards()
        seed_stats = extractSeedStatistics(cards)
        noun_df, verb_df = seed_stats
        fsrs_rows = fetch_fsrs(cards['card_id'].tolist())

        # top_list = top_n_list("es", 7200) #min zipf 4.0
        # top_n_lemmas_with_pos(7200, nlp)

        universe = build_universe(nlp, top_k=50_000, min_zipf=0)

        card_key_map = {}  # dict: card_id -> (lemma, pos)

        # nouns
        for row in noun_df.itertuples(index=False):
            lemma = row.word.lower()
            for cid in row.seed_card_ids:   # <-- list of card_ids
                card_key_map[cid] = (lemma, "NOUN")

        # verbs
        for row in verb_df.itertuples(index=False):
            lemma = row.word.lower()
            for cid in row.seed_card_ids:   # <-- list of card_ids
                card_key_map[cid] = (lemma, "VERB")

        
        fsrs_df = pd.DataFrame(fsrs_rows)

        known_set = build_known_set_from_retrievability(card_key_map, fsrs_df, threshold=0.5, agg="max") #0.9 is my IRL goal

        # report = coverage_report(universe, known_set, auto_known_zipf=4.9)
        rep = coverage_report(
            universe_rows=universe,
            all_cards_lemma_pos_tuples=card_key_map,
            known_set=known_set,
            auto_known_zipf=4.90,
            pos_filter={"NOUN","VERB"},
            decimals=2,
            accent_insensitive=True,
        )

        print(rep)
        # {'summary': 
        # {'universe_count': 22259, 
        # 'all_cards_count_raw': 799, 
        # 'all_cards_count_unique': 785, 
        # 'mastered_count_unique': 439}, 
        # 'zipf_split': {'high_zipf_ge_4': 
        #     {'universe_total': 3177, 
        #     'no_card': 3051, 
        #     'has_card_not_mastered': 46, 
        #     'mastered': 80}, 
        # 'low_zipf_lt_4': 
            # {'universe_total': 19082, 
            # 'no_card': 18846, 
            # 'has_card_not_mastered': 114, 
            # 'mastered': 122}}, 

        # If intersection_deck_exact is low → tagging / normalization work
        # If Zipf 4.5–5.0 coverage is low → make new cards
        # If deck_low_zipf_lt_4_count explodes → rethink card ROI
        # If deck_unmatched_by_zipf_count grows → decide if you want more MWEs or not

        # # universe: list of dicts with keys lemma,pos,zipf
        # universe_keys = {(r["lemma"], r["pos"]) for r in universe}
        # universe_lemmas = {r["lemma"] for r in universe}

        # known_lemmas_only = {lemma for (lemma, pos) in known_set}

        # print("Universe size:", len(universe_keys))
        # print("Known size:", len(known_set))

        # print("Lemma-only intersection:", len(universe_lemmas & known_lemmas_only))
        # print("Exact (lemma,pos) intersection:", len(universe_keys & known_set))
    elif action == 'propose new words by frequency':
        pass
        nlp = spacy.load("es_core_news_sm")

        cards = getAnkiCards()
        seed_stats = extractSeedStatistics(cards)
        noun_df, verb_df = seed_stats

        have_seed = set()
        for row in noun_df.itertuples(index=False):
            have_seed.add((row.word.lower(), "NOUN"))

        for row in verb_df.itertuples(index=False):
            have_seed.add((row.word.lower(), "VERB"))

        # top_list = top_n_list("es", 7200) #min zipf 4.0
        # top_n_lemmas_with_pos(7200, nlp)

        universe = build_universe(nlp, top_k=50_000, min_zipf=4.0)

        card_key_map = {}  # dict: card_id -> (lemma, pos)

        # nouns
        for row in noun_df.itertuples(index=False):
            lemma = row.word.lower()
            for cid in row.seed_card_ids:   # <-- list of card_ids
                card_key_map[cid] = (lemma, "NOUN")

        # verbs
        for row in verb_df.itertuples(index=False):
            lemma = row.word.lower()
            for cid in row.seed_card_ids:   # <-- list of card_ids
                card_key_map[cid] = (lemma, "VERB")


        zipf_lo, zipf_hi = 4.0, 4.9 #I know everything higher :)))

        missing = [
            r for r in universe
            if r["pos"] in {"NOUN", "VERB"}
            and zipf_lo <= r["zipf"] < zipf_hi
            and (r["lemma"], r["pos"]) not in have_seed
        ]
        missing.sort(key=lambda r: r["zipf"], reverse=True)
        
        # print("Top 30:", [(m["lemma"], m["pos"], round(m["zipf"], 2)) for m in missing[:30]])
        for m in missing:
            pass
            # print(m)
        print('Count Missing:'+str(len(missing)))

        lemma_in_sentences = set(
            l for seeds in cards["noun_seeds"].tolist() + cards["verb_seeds"].tolist()
            for l in seeds
        )

        known_lemmas_any_pos = {lemma for (lemma, pos) in have_seed}

        
        for r in missing:
            r["seen_in_sentences"] = r["lemma"] in lemma_in_sentences
            r["known_other_pos"] = r["lemma"] in known_lemmas_any_pos
            r["auto_known"] = r["zipf"] >= 4.8
            if not r["seen_in_sentences"] and not r["known_other_pos"] and not r["auto_known"]:
                print(r)

        counts = Counter(
            (
                r["auto_known"],
                r["seen_in_sentences"],
                r["known_other_pos"],
            )
            for r in missing
        )
        pprint.pprint(counts)
        # Counter({(False, False, False): 2470, #true unknown
        #  (True, False, False): 131, #gave myself credit bc high zipf
        #  (False, False, True): 2}) #found under different POS


        # generate_spanish_word_definition
        deck_name_to_card_dict = {}


        deck_list = []
        for k, v in deck_name_to_card_dict.items():
            deck_list.append(build_anki_deck(k,v))

        num_cards = sum(len(cards) for cards in deck_name_to_card_dict.values())
        print('Num Cards Created: '+str(num_cards))

        ### So, generate an .apkg that is a collect of a few cards for many different decks
        # build_anki_deck(deck_name, deck_dict)

        pkg = genanki.Package(deck_list)
        if DRY_RUN:
            pkg.write_to_file('test_deck.apkg')
            print('Wrote test_deck.apkg')
        else:
            pkg.write_to_file('IRL_deck.apkg')
            print('Wrote IRL_deck.apkg')
        

    elif action == 'inspect revlog':
        con = get_anki_connection(DB_PATH)
        df = get_revlog_df(con)
        print(df.columns)
        print(df.head(1).T)
        
        # Index([
        #     'review_id', --primary key
        #     'card_id', 
        #     'ivl_after', --in days
        #     'ivl_before', --in days
        #     'time_ms', --time spent answering the card in ms
        #     'review_type', -- learn / relearn / review / filtered
        #     'review_dt', --timestamp
        #     'time_s', --time spent answering the card in s
        #     'note_id', 
        #     'deck_id', 
        #     'queue', -- new / learning / review / relearning / suspended ; 0 / 1 / 2 / 3 / -1
        #     'type', -- new / learn / review
        #     'due',  --When the card is due: For reviews: days since collection creation. For learning: seconds offset. Only meaningful relative to Anki’s internal clock.
        #     'reps', --total number of reviews ever
        #     'ivl', --interval, usually matches interval after
        #     'lapses', --failed as a review card
        #     'left', --remaining learning steps
        #     'odue', --Original due date (used when cards are in filtered decks).
        #     'odid', --Original deck ID (also filtered-deck related).
        #     'flds', --Raw note fields, concatenated with \x1f.
        #     'tags', 
        #     'model_id', 
        #     'button'],
        # dtype='object')
        # print(df.head(1).T.to_string())
        # print("Rows:", len(df))

        counts_by_card_id = (
            df.groupby("card_id")
            .size()
            .rename("review_count")
            .sort_values(ascending=False)
            .reset_index()
        )
        print(counts_by_card_id)


        ### List cards I got right the first try in under 30 seconds
        MAX_TIME_S = 30.0
        GOOD_BUTTONS = {"good", "easy"}   # or include "hard" if you consider it "right"

        # ensure chronological order
        df_sorted = df.sort_values(["card_id", "review_id"])
        first_attempt = df_sorted.groupby("card_id", as_index=False).first()

        fast_and_correct_first = first_attempt[
            (first_attempt["time_s"] <= MAX_TIME_S) &
            (first_attempt["button"].isin(GOOD_BUTTONS))
        ]

        ### These I got right on the first attempt and have already been commented out from the source code
        # 5   1769147300799  EspanolActive LearningDave LLM SentencesEstar - Location of People and Objects
        # 24  1769147300895                     EspanolActive LearningDave LLM SentencesComp Coord - porque
        # 35  1769147300965                   EspanolActive LearningDave LLM SentencesComp Coord - mientras
        # 74  1769147301127                          EspanolActive LearningDave LLM SentencesPor - Exchange
        # 75  1769147301135             EspanolActive LearningDave LLM SentencesPor - Communication / Means
        # 76  1769147301139                    EspanolActive LearningDave LLM SentencesPara - Aim / Purpose
        # 79  1769147301147                   EspanolActive LearningDave LLM SentencesPara - Audience / Use
        # 80  1769147301149             EspanolActive LearningDave LLM SentencesPara - Comparison / Opinion
        # 81  1769147301153            EspanolActive LearningDave LLM SentencesPara - Expectation / Outcome

        print('Right first time card ids:')
        print(fast_and_correct_first[["card_id", "deck_name"]].to_string())

    elif action == 'convert 2 sided cards to 1 sided cards':
        pass

    print('Done.')
