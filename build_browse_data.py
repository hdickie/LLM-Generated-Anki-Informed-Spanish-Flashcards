#!/usr/bin/env python3
from __future__ import annotations

import html
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from wordfreq import top_n_list, zipf_frequency

# NLTK WordNet provides access to OMW data when 'omw-1.4' is installed.
from nltk.corpus import wordnet as wn


LANG = "es"
N = 1000
OUTFILE = Path("browse_data_es_top1000.html")


POS_LABEL = {
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "s": "ADJ (sat)",
    "r": "ADV",
}


def infer_pos_from_omw(word: str, lang: str = "spa") -> Tuple[str, str]:
    """
    Infer POS using OMW via NLTK WordNet.

    Strategy:
    - Look up synsets that have this word as a lemma in the given language.
    - Count synset POS tags and choose the most frequent.
    - Return (best_pos_label, debug_counts_string).

    If nothing found, returns ("—", "").
    """
    counts = Counter()

    # wn.synsets(word, lang='es') searches synsets containing that lemma in Spanish
    synsets = wn.synsets(word, lang=lang)
    for s in synsets:
        # WordNet synset.pos() returns: n, v, a, s, r
        counts[s.pos()] += 1

    if not counts:
        return "—", ""

    best_pos, best_count = counts.most_common(1)[0]
    label = POS_LABEL.get(best_pos, best_pos)
    debug = ", ".join(f"{POS_LABEL.get(k,k)}:{v}" for k, v in counts.most_common())
    return label, debug


def build_html(rows: List[Dict[str, str]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Simple static page with client-side search + sort (no libraries)
    # Sort is done by clicking headers; search filters rows.
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Top {N} Spanish Words (Zipf + OMW POS)</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --border: #1f2937;
      --accent: #22c55e;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem 1rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.25rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }}
    h1 {{
      margin: 0 0 0.25rem 0;
      font-size: 1.25rem;
    }}
    .sub {{
      margin: 0 0 1rem 0;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .toolbar {{
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      align-items: center;
      margin: 1rem 0;
    }}
    input[type="search"] {{
      width: min(520px, 100%);
      padding: 0.55rem 0.75rem;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0b1220;
      color: var(--text);
      outline: none;
    }}
    .pill {{
      padding: 0.25rem 0.6rem;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 12px;
    }}
    thead th {{
      text-align: left;
      font-weight: 650;
      color: var(--text);
      background: #0b1220;
      border-bottom: 1px solid var(--border);
      padding: 0.65rem 0.6rem;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}
    tbody td {{
      padding: 0.6rem 0.6rem;
      border-bottom: 1px solid var(--border);
      color: var(--text);
      vertical-align: top;
    }}
    tbody tr:hover td {{
      background: rgba(34,197,94,0.07);
    }}
    .muted {{
      color: var(--muted);
    }}
    .pos {{
      display: inline-block;
      padding: 0.15rem 0.45rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 0.82rem;
    }}
    .note {{
      margin-top: 0.75rem;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.35;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      color: #d1d5db;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Top {N} Spanish words</h1>
      <p class="sub">Generated {now}. Word list: <code>wordfreq.top_n_list(lang="{LANG}")</code>. POS inferred from OMW via NLTK WordNet.</p>

      <div class="toolbar">
        <input id="q" type="search" placeholder="Filter… (matches word or POS)" autocomplete="off"/>
        <span class="pill">Click headers to sort</span>
        <span class="pill" id="count"></span>
      </div>

      <table id="tbl">
        <thead>
          <tr>
            <th data-k="rank">Rank</th>
            <th data-k="word">Word</th>
            <th data-k="zipf">Zipf</th>
            <th data-k="pos">POS</th>
            <th data-k="pos_debug">OMW evidence</th>
          </tr>
        </thead>
        <tbody>
          {''.join(
            f"<tr>"
            f"<td class='muted'>{r['rank']}</td>"
            f"<td><strong>{r['word']}</strong></td>"
            f"<td>{r['zipf']}</td>"
            f"<td><span class='pos'>{r['pos']}</span></td>"
            f"<td class='muted'>{r['pos_debug']}</td>"
            f"</tr>"
            for r in rows
          )}
        </tbody>
      </table>

      <p class="note">
        Notes: POS is a best-effort guess based on how many OMW synsets exist for the lemma in each POS category.
        If a word shows POS as <code>—</code>, OMW didn’t return any synsets for that lemma (or it’s a function word not covered well).
      </p>
    </div>
  </div>

<script>
(function() {{
  const q = document.getElementById('q');
  const tbl = document.getElementById('tbl');
  const tbody = tbl.querySelector('tbody');
  const count = document.getElementById('count');

  function updateCount() {{
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const visible = rows.filter(r => r.style.display !== 'none').length;
    count.textContent = visible + " / {N} shown";
  }}

  function filter() {{
    const needle = (q.value || "").toLowerCase().trim();
    const rows = Array.from(tbody.querySelectorAll('tr'));
    if (!needle) {{
      rows.forEach(r => r.style.display = "");
      updateCount();
      return;
    }}
    rows.forEach(r => {{
      const text = r.innerText.toLowerCase();
      r.style.display = text.includes(needle) ? "" : "none";
    }});
    updateCount();
  }}

  let sortKey = null;
  let sortDir = 1;

  function getCellValue(tr, idx) {{
    const td = tr.children[idx];
    return td ? td.textContent.trim() : "";
  }}

  function sortBy(colIdx, key) {{
    const rows = Array.from(tbody.querySelectorAll('tr'));
    if (sortKey === key) sortDir *= -1;
    else {{ sortKey = key; sortDir = 1; }}

    rows.sort((a,b) => {{
      let av = getCellValue(a, colIdx);
      let bv = getCellValue(b, colIdx);

      // numeric sort for rank and zipf
      if (key === "rank" || key === "zipf") {{
        av = parseFloat(av) || 0;
        bv = parseFloat(bv) || 0;
        return (av - bv) * sortDir;
      }}
      return av.localeCompare(bv) * sortDir;
    }});

    rows.forEach(r => tbody.appendChild(r));
  }}

  tbl.querySelectorAll('thead th').forEach((th, idx) => {{
    th.addEventListener('click', () => sortBy(idx, th.dataset.k));
  }});

  q.addEventListener('input', filter);
  updateCount();
}})();
</script>
</body>
</html>
"""


def main() -> None:
    words = top_n_list(LANG, N)

    rows: List[Dict[str, str]] = []
    for i, w in enumerate(words, start=1):
        z = zipf_frequency(w, LANG)
        pos, debug = infer_pos_from_omw(w, lang="spa")

        rows.append({
            "rank": str(i),
            "word": html.escape(w),
            "zipf": f"{z:.2f}",
            "pos": html.escape(pos),
            "pos_debug": html.escape(debug),
        })

    OUTFILE.write_text(build_html(rows), encoding="utf-8")
    print(f"Wrote: {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
