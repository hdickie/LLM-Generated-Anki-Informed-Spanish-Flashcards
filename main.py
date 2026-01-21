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

def getAnkiCards():
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
                "You are a Spanish sentence generator. "
                "Respond with exactly ONE complete Spanish sentence. "
                "Do NOT include explanations, translations, punctuation outside the sentence, "
                "or any text in any language other than Spanish."
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


if __name__ == "__main__":

    print('Start.')

    action = 'generate cards'
    DB_PATH = "/tmp/collection_ro.anki2"

    if action == 'action1':
        # cp "/Users/hume/Library/Application Support/Anki2/Hume/collection.anki2" /tmp/collection_ro.anki2
        

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

        chatbot_content = (
                "You are a Spanish sentence generator. "
                "Respond with exactly ONE complete Spanish sentence. "
                "Do NOT include explanations, translations, punctuation outside the sentence, "
                "or any text in any language other than Spanish."
            )

        cards = getAnkiCards()
        noun_set = set()
        verb_set = set()
        for _, row in cards.iterrows():
            tags = row.loc['tags']
            if not 'langid:es' in tags:
                continue
            if 'posid:NOUN' in tags:
                noun_set.add(row["front"])
            elif 'posid:VERB' in tags:
                verb_set.add(row["front"])
        
        deck_and_prompt_tuples = []

        ### CLOZE
        # TODO que vs. quien vs. lo que
        # TODO a vs. (no a)
        # TODO uses of se: reflexive and pronominal
        
        # print('SETS')
        # print('Noun: '+str(len(noun_set)))
        # print('Verb: '+str(len(verb_set)))


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
            words_list = noun_set.pop()+', '+verb_set.pop()

            ser_prompt = f"Generate a sentence that uses \"ser\" because of {ser_use_case} and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Ser - {ser_use_case}" , ser_prompt))
        del ser_use_case

        for estar_use_case in ser_use_cases:
            words_list = noun_set.pop()+', '+verb_set.pop()

            estar_prompt = f"Generate a sentence that uses \"estar\" because of {estar_use_case} and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Estar - {estar_use_case}" , estar_prompt))


        se_triggers = [#'Reflexive se','Pronominal', # i feel like this should be handled separately
                       'Reciprocal se', 'Accidental / Unintended se', 'Passive se', 'Impersonal se']
        
        for se_trigger in se_triggers:
            words_list = noun_set.pop()+', '+verb_set.pop()

            se_prompt_preamble_1 = f"Generate a sentence that uses \"se\" because of {se_trigger} and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Se - {se_trigger}" , se_prompt_preamble_1))


        # subjunctive_prompt_preamble_1 = "Generate a sentence that uses the subjunctive trigger phrase ({subj_trigger}) and contains these words or conjugations of these words ({words_list})."

        subj_triggers_1 = ["Volition / Influence","Emotion / Value Judgement","Doubt / Uncertainty",
                         "Purpose / Condition (Time specifically)","Purpose / Condition (Condition specifically)",
                         "Non-existent / indefinite antecedent"]

        for subj_trigger in subj_triggers_1:
            words_list = noun_set.pop()+', '+verb_set.pop()

            subjunctive_prompt_preamble_1 = f"Generate a sentence that uses the subjunctive because of {subj_trigger} and contains these words or conjugations of these words: {words_list}."
            #print(subjunctive_prompt_preamble_1)
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , subjunctive_prompt_preamble_1))

        del subj_trigger
        del words_list
        
        sometimes_subj_triggers = ["cuando","mientras que","hasta que","después de que","tan pronto como"]

        for subj_trigger in sometimes_subj_triggers:
            words_list = noun_set.pop()+', '+verb_set.pop()
            subjunctive_prompt_preamble_2 = f"Generate a sentence that uses the subjunctive trigger phrase \"{subj_trigger}\" because the statement is in the future or not realized and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , subjunctive_prompt_preamble_2))
            words_list = noun_set.pop()+', '+verb_set.pop()
            subjunctive_prompt_preamble_3 = f"Generate a sentence that uses the phrase \"{subj_trigger}\" but does not trigger the subjunctive because the statement is habitual or in the past and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , subjunctive_prompt_preamble_3))
            # print(subjunctive_prompt_preamble_2)
            # print(subjunctive_prompt_preamble_3)

        del subj_trigger
        del words_list

        depends_subj_triggers = ["aunque","aun así","y eso que"]

        for subj_trigger in depends_subj_triggers:
            words_list = noun_set.pop()+', '+verb_set.pop()
            subjunctive_prompt_preamble_4 = f"Generate a sentence that uses the phrase \"{subj_trigger}\" but DOES NOT use the subjunctive because of the intention of the speaker and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , subjunctive_prompt_preamble_4))
            words_list = noun_set.pop()+', '+verb_set.pop()
            subjunctive_prompt_preamble_5 = f"Generate a sentence that uses the phrase \"{subj_trigger}\" and DOES use the subjunctive because of the intention of the speaker and contains these words or conjugations of these words: {words_list}."
            deck_and_prompt_tuples.append((f"Subj - {subj_trigger}" , subjunctive_prompt_preamble_5))
            
            # print(subjunctive_prompt_preamble_4)
            # print(subjunctive_prompt_preamble_5)

        del subj_trigger
        del words_list

        words_list = noun_set.pop()+', '+verb_set.pop()
        subjunctive_prompt_preamble_6 = f"Generate a sentence that uses the imperfect subjunctive and contains these words or conjugations of these words: {words_list}."
        deck_and_prompt_tuples.append((f"Subj - Imperfect" , subjunctive_prompt_preamble_6))
        words_list = noun_set.pop()+', '+verb_set.pop()
        subjunctive_prompt_preamble_7 = f"Generate a sentence that uses the pluperfect subjunctive and contains these words or conjugations of these words: {words_list}."
        deck_and_prompt_tuples.append((f"Subj - Pluperfect" , subjunctive_prompt_preamble_7))

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
            'With que',
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
            'mientras que',
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
                words_list = noun_set.pop()+', '+verb_set.pop()
                comp_coord_prompt = f"Generate a sentence that uses \"{phrase}\" in the \"{comp_coord_cat}\" sense and contains these words or conjugations of these words: {words_list}."
                deck_and_prompt_tuples.append((f"Comp Coord - {phrase}" , comp_coord_prompt))
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
            spanish_sentence_prompt = d_p[1]
            # spanish_sentence = chat(spanish_sentence_prompt)
            # english_sentnece = translate_es_to_en(spanish_sentence)
            spanish_sentence = "Spanish Sentence Placeholder: "+str(spanish_sentence_prompt)
            english_sentence = "English Sentence Placeholder"
            
            print(spanish_sentence)
            print(english_sentence)
            print('-----------------------------------')
            print('')
            # time.sleep(1)

            if not deck_full_name in deck_name_to_card_dict.keys():
                deck_name_to_card_dict[deck_full_name] = {}
            
            deck_name_to_card_dict[deck_full_name][spanish_sentence] = {"back": english_sentence,
                                                                       "tags":[deck_specific_name.replace(" ","_"),"notconfirmed"]}
        del deck_specific_name
        del deck_full_name
        del spanish_sentence_prompt
        del spanish_sentence
        del english_sentence
        
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

            words_list = noun_set.pop()+', '+verb_set.pop()
            cloze_por_prompt = f"Generate a sentence that uses \"por\" exactly once in the sense of \"{por_reason}\" and contains these words or conjugations of these words ({words_list})."
            spanish_sentence = chat(cloze_por_prompt)
            spanish_sentence = spanish_sentence.replace('por','____')
            
            deck_name_to_card_dict[deck_full_name][spanish_sentence] = {"back": deck_specific_name,
                                                                       "tags":[deck_specific_name.replace(" ","_"),"notconfirmed"]}

        del por_reason

        for para_reason in para_reasons:
            deck_specific_name = f"Para - {para_reason}"
            deck_full_name = 'Dave LLM Sentences :: '+deck_specific_name
            if not deck_full_name in deck_name_to_card_dict.keys():
                deck_name_to_card_dict[deck_full_name] = {}

            words_list = noun_set.pop()+', '+verb_set.pop()
            cloze_para_prompt = f"Generate a sentence that uses \"para\" exactly once in the sense of \"{para_reason}\" and contains these words or conjugations of these words ({words_list})."
            spanish_sentence = chat(cloze_para_prompt)
            spanish_sentence = spanish_sentence.replace('para','____')
            
            deck_name_to_card_dict[deck_full_name][spanish_sentence] = {"back": deck_specific_name,
                                                                       "tags":[deck_specific_name.replace(" ","_"),"notconfirmed"]}
        del para_reason
        
        
        deck_list = []
        for k, v in deck_name_to_card_dict.items():
            deck_list.append(build_anki_deck(k,v))

        ### So, generate an .apkg that is a collect of a few cards for many different decks
        # build_anki_deck(deck_name, deck_dict)

        pkg = genanki.Package(deck_list)
        pkg.write_to_file('test_deck.apkg')
        print('Wrote test_deck.apkg')


    print('Done.')
