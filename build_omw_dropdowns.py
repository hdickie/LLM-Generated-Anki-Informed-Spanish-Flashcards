#!/usr/bin/env python3
from __future__ import annotations

import html
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from wordfreq import top_n_list, zipf_frequency
from nltk.corpus import wordnet as wn

# wordfreq uses ISO-639-1; WordNet/OMW uses ISO-639-3
WORDFREQ_LANG = "es"
OMW_LANG = "spa"

N = 1000
OUTFILE = Path("browse_omw_dropdowns_es_top1000.html")

POS_LABEL = {
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "s": "ADJ (sat)",
    "r": "ADV",
}
POS_ORDER = ["NOUN", "VERB", "ADJ", "ADV", "ADJ (sat)", "—"]


def synsets_for_word(word: str) -> List[wn.synset]:
    return wn.synsets(word, lang=OMW_LANG)


def best_pos_label(word: str) -> Tuple[str, Counter]:
    counts = Counter()
    for s in synsets_for_word(word):
        counts[s.pos()] += 1
    if not counts:
        return "—", counts
    best_pos = counts.most_common(1)[0][0]
    return POS_LABEL.get(best_pos, best_pos), counts


def spanish_lemmas_for_synset(s) -> List[str]:
    # Get Spanish lemma names for this synset (OMW)
    names = []
    for lemma in s.lemmas(lang=OMW_LANG):
        n = lemma.name().replace("_", " ")
        names.append(n)
    # unique, stable order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_html(grouped: Dict[str, List[Dict]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Simple inline CSS; no JS needed.
    css = """
    :root { --bg:#fafafa; --fg:#111827; --muted:#6b7280; --border:#e5e7eb; --card:#fff; }
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--fg); }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding: 1.25rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    h1 { margin:0 0 .25rem 0; font-size: 1.25rem; }
    .sub { margin:0 0 1rem 0; color:var(--muted); }
    details { border: 1px solid var(--border); border-radius: 12px; padding: .65rem .75rem; background: white; }
    details + details { margin-top: .6rem; }
    summary { cursor: pointer; font-weight: 650; }
    .meta { color: var(--muted); font-weight: 500; margin-left:.35rem; }
    .word { margin-top:.6rem; }
    .word summary { font-weight: 600; }
    .kv { margin: .35rem 0 .6rem 0; color: var(--muted); font-size: .95rem; }
    .synset { margin:.45rem 0 .65rem 0; padding:.55rem .6rem; border:1px solid var(--border); border-radius:12px; background:#fcfcfd; }
    .synset .head { font-weight: 650; }
    .synset .def { margin:.25rem 0 0 0; color: var(--fg); }
    .synset .lemmas { margin:.35rem 0 0 0; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    """

    # Build POS sections
    pos_sections = []
    for pos in POS_ORDER:
        words = grouped.get(pos, [])
        if not words:
            continue

        # build word blocks
        word_blocks = []
        for w in words:
            # each word is a dropdown
            syn_blocks = []
            for s in w["synsets"]:
                syn_id = html.escape(s["id"])
                syn_pos = html.escape(s["pos"])
                syn_def = html.escape(s["def"])
                lemmas = html.escape(", ".join(s["es_lemmas"])) if s["es_lemmas"] else "—"
                syn_blocks.append(
                    f"""
                    <div class="synset">
                      <div class="head">{syn_id} <span class="meta">{syn_pos}</span></div>
                      <p class="def">{syn_def}</p>
                      <div class="lemmas">es lemmas: {lemmas}</div>
                    </div>
                    """.strip()
                )

            counts_str = html.escape(w["pos_counts"]) if w["pos_counts"] else ""
            word_blocks.append(
                f"""
                <details class="word">
                  <summary>{html.escape(w["word"])} <span class="meta">Zipf {w["zipf"]} · {w["synset_count"]} synsets</span></summary>
                  <div class="kv">OMW POS evidence: <code>{counts_str or "—"}</code></div>
                  {''.join(syn_blocks) if syn_blocks else '<div class="kv">No OMW synsets found.</div>'}
                </details>
                """.strip()
            )

        pos_sections.append(
            f"""
            <details>
              <summary>{pos} <span class="meta">({len(words)} words)</span></summary>
              <div style="margin-top:.65rem;">
                {''.join(word_blocks)}
              </div>
            </details>
            """.strip()
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>OMW Browser (Spanish Top {N})</title>
  <style>{css}</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>OMW browser: Spanish top {N} words</h1>
      <p class="sub">
        Generated {now}. Frequency list via <code>wordfreq.top_n_list("{WORDFREQ_LANG}")</code>.
        OMW lookup via <code>nltk.corpus.wordnet</code> with <code>lang="{OMW_LANG}"</code>.
      </p>

      {''.join(pos_sections)}

      <p class="sub" style="margin-top:1rem;">
        Tip: use your browser’s find (Ctrl/Cmd+F) to jump to a word quickly.
      </p>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    words = top_n_list(WORDFREQ_LANG, N)

    grouped: Dict[str, List[Dict]] = defaultdict(list)

    for w in words:
        z = zipf_frequency(w, WORDFREQ_LANG)
        pos_label, counts = best_pos_label(w)

        # pos evidence string like "NOUN:3, VERB:1"
        if counts:
            pos_counts = ", ".join(f"{POS_LABEL.get(k,k)}:{v}" for k, v in counts.most_common())
        else:
            pos_counts = ""

        synsets = []
        for s in synsets_for_word(w):
            synsets.append({
                "id": s.name(),                  # e.g. "house.n.01"
                "pos": POS_LABEL.get(s.pos(), s.pos()),
                "def": s.definition(),           # typically English (WordNet)
                "es_lemmas": spanish_lemmas_for_synset(s),
            })

        grouped[pos_label].append({
            "word": w,
            "zipf": f"{z:.2f}",
            "synset_count": len(synsets),
            "pos_counts": pos_counts,
            "synsets": synsets,
        })

    html_out = build_html(grouped)
    OUTFILE.write_text(html_out, encoding="utf-8")
    print(f"Wrote: {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
