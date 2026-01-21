#https://cvc.cervantes.es/ensenanza/biblioteca_ele/plan_curricular/indice.htm
from __future__ import annotations
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

import sqlite3
import pandas as pd

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

    return pd.DataFrame(card_data)


def main():
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
        LIMIT 20
    """).fetchall()

    for r in rows:
        mid = int(r["mid"])
        fieldnames = fieldmap.get(mid, [])
        front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

        out = {
            "deck": decks.get(int(r["did"]), f"Unknown(did={r['did']})"),
            "front": front,
            "back": back,
            "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
            "due": due_display(int(r["queue"]), int(r["due"]), col_crt),
            "tags": (r["tags"] or "").strip().split(),
        }
        print(out)

    con.close()

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

class SeedPicker:
    def __init__(self, usage_stats, rng_seed=0):
        self.df = usage_stats.copy()
        self.rng = np.random.default_rng(rng_seed)

        for col in ["new_count", "young_count", "mature_count", "total_count"]:
            if col not in self.df.columns:
                self.df[col] = 0

    def _score(self, row):
        return (
            row["total_count"]
            + 3 * row["new_count"]
            + 2 * row["young_count"]
            + 0.5 * row["mature_count"]
        )

    def next(self):
        scores = self.df.apply(self._score, axis=1).to_numpy()
        K = min(50, len(self.df))
        best_idx = np.argpartition(scores, K - 1)[:K]

        weights = 1.0 / (scores[best_idx] + 1e-6)
        weights /= weights.sum()

        i = self.rng.choice(best_idx, p=weights)
        word = self.df.iloc[i]["word"]

        # optimistic update
        self.df.at[self.df.index[i], "new_count"] += 1
        self.df.at[self.df.index[i], "total_count"] += 1

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


import pandas as pd

MATURE_IVL_DAYS_DEFAULT = 21  # tune if you want (Anki “mature” is commonly 21d+)

def extractSeedStatistics(
    cards_df: pd.DataFrame,
    *,
    mature_ivl_days: int = MATURE_IVL_DAYS_DEFAULT,
    # sentence_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build usage stats for noun/verb seeds from your cards DF.

    Returns:
      (noun_usage_stats_df, verb_usage_stats_df)

    Output schema (both DFs):
      word, total_count, new_count, young_count, mature_count
      (+ learning_count, suspended_count, buried_count as extra helpful columns)

    Notes on classification (based on Anki card queue/ivl):
      - new_count: queue == 0
      - mature_count: queue == 2 and ivl >= mature_ivl_days
      - young_count: everything else "active-ish" (review young + learning),
        i.e. (queue == 2 and ivl < mature_ivl_days) OR (queue in {1,3})
      - suspended/buried tracked separately
    """

    df = cards_df.copy()

    # ---------- Decide which cards count as "SENTENCE cards" ----------
    # if sentence_only:
    #     if "is_sentence_card" in df.columns:
    #         df = df[df["is_sentence_card"] == True]
    #     else:
    # Heuristic fallback: include only notes that actually have seed tags
    # (this is usually what you want anyway)

    # print('DF')
    # print(df)

    if "tags" in df.columns:
        def _has_seed_tag(ts):
            # print("TAGS:", ts)
            if not ts:
                return False
            for t in ts:
                # print("  checking tag:", t)
                if t.startswith("posid:VERB") or t.startswith("posid:NOUN"):
                    # print("  -> MATCH")
                    return True
            # print("  -> NO MATCH")
            return False

        df = df[df["tags"].apply(_has_seed_tag)]

    else:
        raise ValueError("cards_df must include 'tags' to filter sentence cards.")

    # Make sure front is a clean "seed" string
    df["front_seed"] = df["front"].fillna("").astype(str).str.strip()

    def _has_tag(ts, tag):
        return bool(ts) and (tag in ts)

    # noun_seeds / verb_seeds become either [front_word] or []
    df["noun_seeds"] = df.apply(
        lambda r: [r["front_seed"]] if _has_tag(r["tags"], "posid:NOUN") and r["front_seed"] else [],
        axis=1,
    )

    df["verb_seeds"] = df.apply(
        lambda r: [r["front_seed"]] if _has_tag(r["tags"], "posid:VERB") and r["front_seed"] else [],
        axis=1,
    )

    # print(df["verb_seeds"])

    # ---------- Normalize required scheduling fields ----------
    for col in ["queue", "ivl"]:
        if col not in df.columns:
            raise ValueError(f"cards_df is missing required column: '{col}'")
    df["queue"] = df["queue"].fillna(0).astype(int)
    df["ivl"] = df["ivl"].fillna(0).astype(int)

    # ---------- Classification flags ----------
    # Anki queue reference (common):
    #  0=new, 1=learning, 2=review, 3=day learn, -1=suspended, -2=buried
    df["is_new"]       = (df["queue"] == 0)
    df["is_review"]    = (df["queue"] == 2)
    df["is_learning"]  = df["queue"].isin([1, 3])
    df["is_suspended"] = (df["queue"] == -1)
    df["is_buried"]    = (df["queue"] == -2)

    df["is_mature"] = df["is_review"] & (df["ivl"] >= mature_ivl_days)
    df["is_young"]  = (df["is_review"] & (df["ivl"] < mature_ivl_days)) | df["is_learning"]

    # ---------- Helper to build per-seed stats ----------
    def _stats_for_seedcol(seed_col: str) -> pd.DataFrame:
        tmp = df[["card_id", seed_col, "is_new", "is_young", "is_mature", "is_learning", "is_suspended", "is_buried"]].copy()

        # One row per (card, seed) — if you ever allow multiple seeds per note, this handles it.
        tmp = tmp.explode(seed_col, ignore_index=True)
        tmp = tmp.rename(columns={seed_col: "word"})
        tmp = tmp[tmp["word"].notna() & (tmp["word"].astype(str).str.len() > 0)]

        # Aggregate: counts per word
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
                )
        )

        # Convenience: sort by “load” so least-used floats to top
        out["load_score"] = (
            out["total_count"]
            + 3.0 * out["new_count"]
            + 2.0 * out["young_count"]
            + 0.5 * out["mature_count"]
        )
        out = out.sort_values(["load_score", "total_count", "word"], ascending=[True, True, True]).reset_index(drop=True)
        return out

    noun_usage_stats = _stats_for_seedcol("noun_seeds")
    verb_usage_stats = _stats_for_seedcol("verb_seeds")

    return noun_usage_stats, verb_usage_stats



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
    Build usage stats per learning objective from a cards DF.

    Expects cards_df columns:
      - 'tags' (list[str] or space-delimited string)
      - 'queue' (int), 'ivl' (int)
      - optionally 'card_id'

    Returns DF with:
      objective, total_count, new_count, young_count, mature_count,
      learning_count, suspended_count, buried_count, load_score
    """

    df = cards_df.copy()

    # Normalize tags into list[str]
    def _norm_tags(ts):
        if ts is None:
            return []
        if isinstance(ts, str):
            return ts.split()
        return list(ts)

    df["tags_norm"] = df["tags"].apply(_norm_tags)

    # Extract objectives from tags: learningobjective:OBJECTIVE
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
            "objective", "total_count", "new_count", "young_count", "mature_count",
            "learning_count", "suspended_count", "buried_count", "load_score"
        ])

    # Ensure scheduling fields exist
    if "queue" not in df.columns or "ivl" not in df.columns:
        raise ValueError("cards_df must include 'queue' and 'ivl' columns.")

    df["queue"] = df["queue"].fillna(0).astype(int)
    df["ivl"] = df["ivl"].fillna(0).astype(int)

    # Classification
    df["is_new"]       = (df["queue"] == 0)
    df["is_review"]    = (df["queue"] == 2)
    df["is_learning"]  = df["queue"].isin([1, 3])
    df["is_suspended"] = (df["queue"] == -1)
    df["is_buried"]    = (df["queue"] == -2)

    df["is_mature"] = df["is_review"] & (df["ivl"] >= mature_ivl_days)
    df["is_young"]  = (df["is_review"] & (df["ivl"] < mature_ivl_days)) | df["is_learning"]

    # Choose an id column
    id_col = "card_id" if "card_id" in df.columns else None
    if id_col is None:
        df = df.reset_index().rename(columns={"index": "_row_id"})
        id_col = "_row_id"

    # Explode objectives
    tmp = df[[id_col, "objectives", "is_new", "is_young", "is_mature",
              "is_learning", "is_suspended", "is_buried"]].copy()

    tmp = (
        tmp.explode("objectives", ignore_index=True)
           .rename(columns={"objectives": "objective"})
    )

    tmp = tmp[tmp["objective"].notna() & (tmp["objective"].astype(str).str.len() > 0)]

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
           )
    )

    out["load_score"] = (
        out["total_count"]
        + 3.0 * out["new_count"]
        + 2.0 * out["young_count"]
        + 0.5 * out["mature_count"]
    )

    out = out.sort_values(
        ["load_score", "total_count", "objective"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return out


if __name__ == "__main__":

    print('Start.')

    action = 'generate cards'
    DB_PATH = "/tmp/collection_ro.anki2"

    DRY_RUN = True
    SHOW_CARDS = True

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
            'Location of People and Objects',
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
            'porque',
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
            'mientras',
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
            'dicho de otro modo'
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
            # spanish_sentence = chat(spanish_sentence_prompt)
            # english_sentnece = translate_es_to_en(spanish_sentence)
            spanish_sentence = "Spanish Sentence Placeholder: "+str(spanish_sentence_prompt)
            english_sentence = "English Sentence Placeholder: "+str(spanish_sentence_prompt)
            
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
            "Exchange",
            "Reason / Motive",
            "Frequency / Duration",
            "Emotions / Opinions",
            "Communication / Means",
            "Travel / Through"]
        para_reasons = [
            "Aim / Purpose",
            "Target / Destination",
            "Time limit / Deadline",
            "Recipient",
            "Audience / Use",
            "Comparison / Opinion",
            "Toward (movement)",
            "Expectation / Outcome",
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
            spanish_sentence = re.sub(r'\b[Pp]or\b', '____', spanish_sentence, count=1)
            
            deck_name_to_card_dict[deck_full_name][spanish_sentence] = {"back": deck_specific_name,
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
            spanish_sentence = re.sub(r'\b[Pp]ara\b', '____', spanish_sentence, count=1)

            
            deck_name_to_card_dict[deck_full_name][spanish_sentence] = {"back": deck_specific_name,
                                                                       "tags":["learningobjective:"+deck_specific_name.replace(" ","_"),
                                                                               "notconfirmed",
                                                                                "nounseed:"+noun,
                                                                                "verbseed:"+verb
                                                                                ]}
        del para_reason
        del noun
        del verb
        
        #TODO here we can iterate over deck_name_to_card_dict and reduce output as needed

        objective_stats = extractObjectiveStatistics(cards)
        print(objective_stats.to_string())
        
        deck_list = []
        for k, v in deck_name_to_card_dict.items():
            deck_list.append(build_anki_deck(k,v))

        num_cards = sum(len(cards) for cards in deck_name_to_card_dict.values())
        print('Num Cards Created: '+str(num_cards))

        ### So, generate an .apkg that is a collect of a few cards for many different decks
        # build_anki_deck(deck_name, deck_dict)

        pkg = genanki.Package(deck_list)
        pkg.write_to_file('test_deck.apkg')
        print('Wrote test_deck.apkg')
        # print('noun_usage_stats:')
        # print(noun_usage_stats)
        # print('verb_usage_stats:')
        # print(verb_usage_stats)


    print('Done.')
