#!/usr/bin/env python3
from __future__ import annotations

import html
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from nltk.corpus import wordnet as wn

OUTFILE = Path("omw_es_by_lexname.html")
OMW_LANG = "spa"  # WordNet/OMW expects ISO-639-3, not "es"


def uniq_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def es_lemmas_for_synset(s) -> List[str]:
    # Spanish lemma names for this synset
    names = [l.name().replace("_", " ") for l in s.lemmas(lang=OMW_LANG)]
    return uniq_preserve(names)


def build_html(grouped: Dict[str, Dict[str, List[Dict]]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    css = """
    :root {
      --bg:#fafafa; --fg:#111827; --muted:#6b7280; --border:#e5e7eb; --card:#fff;
    }
    body { margin:0; font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.25rem; box-shadow:0 1px 2px rgba(0,0,0,0.05); }
    h1 { margin:0 0 .25rem 0; font-size:1.25rem; }
    .sub { margin:0 0 1rem 0; color:var(--muted); max-width: 85ch; }
    details { border:1px solid var(--border); border-radius: 12px; padding: .65rem .75rem; background:white; }
    details + details { margin-top: .6rem; }
    summary { cursor:pointer; font-weight:650; }
    .meta { color:var(--muted); font-weight:500; margin-left:.35rem; }
    .synset { margin:.55rem 0; padding:.6rem .65rem; border:1px solid var(--border); border-radius: 12px; background:#fcfcfd; }
    .synset .head { font-weight:650; }
    .synset .def { margin:.25rem 0 0 0; }
    .lemmas { margin:.35rem 0 0 0; color:var(--muted); font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace; }
    code { font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace; }
    """

    # Stable POS order
    pos_order = ["n", "v", "a", "r", "s"]
    pos_name = {"n":"NOUN", "v":"VERB", "a":"ADJ", "r":"ADV", "s":"ADJ (sat)"}

    pos_blocks = []
    for p in pos_order:
        lexmap = grouped.get(p, {})
        if not lexmap:
            continue

        # Sort lexnames (noun.food, noun.state, ...)
        lexnames = sorted(lexmap.keys())
        lex_blocks = []

        for lex in lexnames:
            syns = lexmap[lex]
            # Sort synsets by number of Spanish lemmas desc, then by synset name
            syns_sorted = sorted(syns, key=lambda d: (-len(d["es_lemmas"]), d["name"]))

            syn_blocks = []
            for d in syns_sorted:
                syn_blocks.append(
                    f"""
                    <details class="synset">
                      <summary>{html.escape(d["name"])} <span class="meta">({len(d["es_lemmas"])} es lemmas)</span></summary>
                      <p class="def">{html.escape(d["definition"] or "—")}</p>
                      <div class="lemmas">es: {html.escape(", ".join(d["es_lemmas"]) or "—")}</div>
                    </details>
                    """.strip()
                )

            lex_blocks.append(
                f"""
                <details>
                  <summary>{html.escape(lex)} <span class="meta">({len(syns)} synsets)</span></summary>
                  <div style="margin-top:.65rem;">
                    {''.join(syn_blocks)}
                  </div>
                </details>
                """.strip()
            )

        pos_blocks.append(
            f"""
            <details>
              <summary>{pos_name.get(p,p)} <span class="meta">({sum(len(lexmap[k]) for k in lexmap)} synsets)</span></summary>
              <div style="margin-top:.65rem;">
                {''.join(lex_blocks)}
              </div>
            </details>
            """.strip()
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>OMW Spanish Browser by Lexname</title>
  <style>{css}</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>OMW Spanish browser — grouped by WordNet lexname</h1>
      <p class="sub">
        Generated {now}. This groups WordNet synsets that have Spanish lemmas (<code>lang="{OMW_LANG}"</code>)
        by <code>synset.lexname()</code> (e.g. <code>noun.food</code>, <code>noun.state</code>).
        Definitions are WordNet’s (typically English).
      </p>

      {''.join(pos_blocks)}

      <p class="sub" style="margin-top:1rem;">
        Tip: use your browser’s find (Cmd/Ctrl+F) for quick jumps (e.g. search <code>noun.food</code> or a Spanish lemma).
      </p>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    # grouped[pos][lexname] -> list of synset dicts
    grouped: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    # NOTE: all_synsets() is big, but for a one-time dev export it's fine.
    for s in wn.all_synsets():
        # Only keep synsets that have *any* Spanish lemmas in OMW
        es_lemmas = es_lemmas_for_synset(s)
        if not es_lemmas:
            continue

        grouped[s.pos()][s.lexname()].append({
            "name": s.name(),                 # e.g., "food.n.01"
            "definition": s.definition(),     # typically English
            "es_lemmas": es_lemmas,
        })

    html_out = build_html(grouped)
    OUTFILE.write_text(html_out, encoding="utf-8")
    print(f"Wrote: {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
