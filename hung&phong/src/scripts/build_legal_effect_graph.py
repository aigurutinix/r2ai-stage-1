"""Build a curated legal-effect graph artifact for ranking/selection.

This graph is intentionally conservative.  It combines:

- existing project knowledge from v66/v49 version maps;
- document metadata graph generated from Qdrant;
- benchmark cutoff awareness.

It is not a substitute for an official legal-effect database, but it makes the
pipeline's version reasoning explicit and data-driven instead of scattered in
code constants.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


OBSOLETE_PREFIXES = {
    "03/2003", "10/2012", "28/2011", "33/2005", "43/2010", "44/2003",
    "60/2005", "68/2014", "78/2006", "78/2015", "83/2013", "85/2007",
    "156/2013", "166/2013", "198-CP", "05/2015", "28/2020", "37/2006",
    "06/2006", "24/2007", "47/2010", "113/2004", "164/2003", "123/2012",
    "78/2014", "109/2004", "39/2018", "101/2001", "102/2001", "154/2005",
    # v49 known-old additions.
    "67/2014", "59/2005", "70/2006", "05/2010", "03/2014", "148/2018",
    "27/2014", "169/2011", "211/2013", "92/2015", "20/2015", "05/2013",
    "96/2015", "51/2010", "04/2014", "119/2018", "39/2014", "32/2011",
    "35/2006", "154/2005",
}

SUCCESSORS = {
    "03/2003": ["88/2015"],
    "10/2012": ["45/2019"],
    "28/2011": ["80/2021", "38/2019", "108/2025"],
    "33/2005": ["91/2015"],
    "43/2010": ["168/2025", "01/2021"],
    "44/2003": ["45/2019", "145/2020"],
    "60/2005": ["59/2020"],
    "68/2014": ["59/2020"],
    "78/2006": ["38/2019", "108/2025"],
    "78/2015": ["168/2025", "01/2021"],
    "83/2013": ["126/2020", "80/2021", "38/2019"],
    "85/2007": ["38/2019", "108/2025"],
    "156/2013": ["80/2021", "38/2019"],
    "166/2013": ["125/2020"],
    "28/2020": ["12/2022"],
    "47/2010": ["12/2022", "45/2019"],
    "113/2004": ["12/2022", "45/2019"],
    "109/2004": ["168/2025"],
    "39/2018": ["80/2021"],
    "101/2001": ["54/2014"],
    "102/2001": ["54/2014"],
    "154/2005": ["54/2014"],
    "67/2014": ["61/2020"],
    "59/2005": ["61/2020"],
    "70/2006": ["54/2019"],
    "05/2015": ["145/2020"],
    "05/2010": ["12/2022", "45/2019"],
    "03/2014": ["12/2022", "45/2019"],
    "148/2018": ["145/2020"],
    "27/2014": ["45/2019"],
    "169/2011": ["38/2019", "108/2025"],
    "211/2013": ["38/2019", "108/2025"],
    "92/2015": ["38/2019", "108/2025"],
    "20/2015": ["168/2025", "01/2021"],
    "05/2013": ["168/2025", "01/2021"],
    "96/2015": ["168/2025", "01/2021"],
    "51/2010": ["123/2020"],
    "04/2014": ["123/2020"],
    "119/2018": ["123/2020"],
    "39/2014": ["123/2020"],
    "32/2011": ["123/2020"],
    "37/2006": ["81/2018", "26/VBHN-BCT"],
    "06/2006": ["09/2018"],
}

CURRENT_BOOSTS = {
    "12/2022": 0.10,
    "38/2019": 0.08,
    "45/2019": 0.10,
    "59/2020": 0.12,
    "80/2021": 0.08,
    "88/2015": 0.08,
    "91/2015": 0.10,
    "108/2025": 0.12,
    "125/2020": 0.10,
    "168/2025": 0.16,
    "54/2010": 0.05,
    "54/2014": 0.06,
    "50/2005": 0.04,
    "36/2005": 0.04,
}

STILL_CURRENT_OLD_PREFIXES = {
    "36/2005",
    "50/2005",
    "54/2010",
    "54/2014",
    "88/2015",
    "91/2015",
}


def main() -> None:
    doc_graph_path = ROOT / "data" / "doc_metadata_graph_20260630.json"
    doc_graph = json.loads(doc_graph_path.read_text(encoding="utf-8")) if doc_graph_path.exists() else {}
    successors = {k: list(v) for k, v in SUCCESSORS.items()}
    for old, news in (doc_graph.get("successors_from_title_refs") or {}).items():
        cur = successors.setdefault(old, [])
        for n in news:
            if n not in cur:
                cur.append(n)

    out = {
        "eval_cutoff": doc_graph.get("eval_cutoff", "2026-03-31"),
        "obsolete_prefixes": sorted(OBSOLETE_PREFIXES),
        "still_current_old_prefixes": sorted(STILL_CURRENT_OLD_PREFIXES),
        "successors": dict(sorted((k, sorted(set(v))) for k, v in successors.items())),
        "current_boosts": dict(sorted(CURRENT_BOOSTS.items())),
        "after_eval_cutoff_prefixes": sorted(
            k for k, d in (doc_graph.get("docs") or {}).items() if d.get("after_eval_cutoff")
        ),
        "notes": [
            "Curated + title-inferred graph. Use for ranking evidence, not as official legal status.",
            "still_current_old_prefixes prevents blanket pre-2015 suppression for important still-current laws.",
            "after_eval_cutoff_prefixes should be penalized for benchmark submissions unless explicitly asked.",
        ],
    }
    path = ROOT / "data" / "legal_effect_graph_20260630.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(path),
        "obsolete": len(out["obsolete_prefixes"]),
        "successor_sources": len(out["successors"]),
        "current_boosts": len(out["current_boosts"]),
        "after_cutoff": out["after_eval_cutoff_prefixes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
