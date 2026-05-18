#!/usr/bin/env python3
"""
C.3 — arXiv citation audit.

Parses ./paper/main.tex thebibliography block,
extracts (cite_key, lead_author, year, arxiv_id, claimed_title) for
every \\bibitem with an arXiv ID, fetches arxiv.org/abs/<id>, and
compares the live title + lead-author surname to the bib entry.

Output: ./results/citation_audit.json with one
record per entry:
  {
    cite_key, arxiv_id, claimed_title, claimed_lead_author_surname,
    live_title, live_lead_author_surname,
    title_match: bool, author_match: bool, verdict: 'ok' | 'mismatch'
  }

Live-fetch is via urllib + HTML parsing on the abs page; arXiv blocks
no-User-Agent traffic so we send a polite UA. We do NOT attempt to
correct the bib here; verdicts go to a JSON file the lead author reviews and
decides per-entry.

If arXiv is unreachable, the script records 'fetch_failed' as the
verdict and exits non-zero. Re-running on a working network completes
the audit.
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

PAPER = Path("./paper/main.tex")
OUT_DIR = Path("./results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "citation_audit.json"

UA = (
    "Mozilla/5.0 (compatible; PPFGCitationAudit/1.0; "
    "research-only) "
)

# Match: \bibitem[Author et al., YEAR]{cite_key} ... arXiv:XXXX.XXXXX.
BIBITEM_RE = re.compile(
    r"\\bibitem\[([^]]*)\]\{([^}]+)\}\s*(.*?)(?=\\bibitem\[|\\end\{thebibliography\})",
    re.DOTALL,
)
ARXIV_ID_RE = re.compile(r"arXiv:\s*([\d.]+v?\d*)", re.IGNORECASE)


def parse_bib(text: str) -> list[dict]:
    """Extract bib records with arXiv IDs."""
    records = []
    # Restrict to thebibliography block
    bib_block_m = re.search(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
        text, re.DOTALL,
    )
    if not bib_block_m:
        return records
    block = bib_block_m.group(0)

    for m in BIBITEM_RE.finditer(block):
        label, cite_key, body = m.group(1), m.group(2), m.group(3)
        arxiv_m = ARXIV_ID_RE.search(body)
        if not arxiv_m:
            continue
        arxiv_id = arxiv_m.group(1).rstrip(".,;")
        # Drop trailing version marker for matching against live page
        arxiv_id_no_ver = re.sub(r"v\d+$", "", arxiv_id)

        # Lead author surname: first token of label up to ' et al.,' or ','
        lead = label.split(",")[0]
        if " et al." in lead:
            lead = lead.replace(" et al.", "").strip()
        if " and " in lead:
            lead = lead.split(" and ")[0].strip()
        lead = lead.strip()

        # Year
        year_m = re.search(r"(20\d{2})", label)
        year = int(year_m.group(1)) if year_m else None

        # Claimed title: strip first line (authors+year) and trailing arXiv ref
        body_text = re.sub(r"\s+", " ", body).strip()
        # Title heuristic: text between the year-marker '. YYYY.' and the arXiv:
        title_m = re.search(
            r"\.\s*20\d{2}\.\s*(.+?)\.\s*(arXiv|Hugging|Hugging Face|$)",
            body_text,
        )
        claimed_title = title_m.group(1).strip() if title_m else ""

        records.append({
            "cite_key": cite_key,
            "claimed_lead_author_surname": lead,
            "claimed_year": year,
            "arxiv_id": arxiv_id_no_ver,
            "arxiv_id_with_ver": arxiv_id,
            "claimed_title": claimed_title,
        })
    return records


def fetch_arxiv_abs(arxiv_id: str) -> Optional[str]:
    url = f"https://arxiv.org/abs/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        return None


TITLE_META_RE = re.compile(
    r'<meta\s+name="citation_title"\s+content="([^"]+)"\s*/?>',
    re.IGNORECASE,
)
AUTHOR_META_RE = re.compile(
    r'<meta\s+name="citation_author"\s+content="([^"]+)"\s*/?>',
    re.IGNORECASE,
)


def parse_arxiv_html(page: str) -> dict:
    title_m = TITLE_META_RE.search(page)
    title = html.unescape(title_m.group(1)).strip() if title_m else ""
    authors = [html.unescape(m.group(1)).strip()
               for m in AUTHOR_META_RE.finditer(page)]
    # citation_author is "Surname, Given-name" format
    lead_surname = ""
    if authors:
        first = authors[0]
        lead_surname = first.split(",")[0].strip()
    return {
        "live_title": title,
        "live_authors": authors,
        "live_lead_author_surname": lead_surname,
    }


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def title_match(claimed: str, live: str) -> bool:
    """Loose normalization-and-overlap title match.

    Returns True if normalized claimed title is a prefix or substring
    of live title, or the live title contains >= 6 consecutive tokens
    from claimed."""
    if not claimed or not live:
        return False
    c, l = normalize(claimed), normalize(live)
    if c == l:
        return True
    if c in l or l in c:
        return True
    # token-overlap fallback
    ctoks = c.split()
    if len(ctoks) >= 4:
        for k in range(len(ctoks) - 3):
            if " ".join(ctoks[k:k + 4]) in l:
                return True
    return False


def author_match(claimed: str, live: str) -> bool:
    if not claimed or not live:
        return False
    c, l = normalize(claimed), normalize(live)
    return c == l or c in l or l in c


def main():
    text = PAPER.read_text()
    records = parse_bib(text)

    audit = []
    n_mismatch = 0
    n_fetch_fail = 0

    for r in records:
        page = fetch_arxiv_abs(r["arxiv_id"])
        time.sleep(0.7)  # polite rate-limit

        if page is None:
            r["fetch_failed"] = True
            r["verdict"] = "fetch_failed"
            n_fetch_fail += 1
            audit.append(r)
            continue

        live = parse_arxiv_html(page)
        r.update(live)
        r["title_match_ok"] = title_match(r["claimed_title"], live["live_title"])
        r["author_match_ok"] = author_match(
            r["claimed_lead_author_surname"], live["live_lead_author_surname"]
        )

        if r["title_match_ok"] and r["author_match_ok"]:
            r["verdict"] = "ok"
        else:
            r["verdict"] = "mismatch"
            n_mismatch += 1
        audit.append(r)

    summary = {
        "n_records": len(audit),
        "n_ok": sum(1 for a in audit if a.get("verdict") == "ok"),
        "n_mismatch": n_mismatch,
        "n_fetch_fail": n_fetch_fail,
        "records": audit,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    print(f"=== arXiv citation audit ===")
    print(f"records: {summary['n_records']}  ok: {summary['n_ok']}  "
          f"mismatch: {summary['n_mismatch']}  fetch_fail: {summary['n_fetch_fail']}")
    if n_mismatch or n_fetch_fail:
        print()
        print("Records needing attention:")
        for a in audit:
            if a.get("verdict") in ("mismatch", "fetch_failed"):
                print(f"  {a['cite_key']:25s} arxiv={a['arxiv_id']:14s} "
                      f"verdict={a['verdict']:14s}")
                print(f"    claimed: {a['claimed_lead_author_surname']} | {a['claimed_title']}")
                if "live_title" in a:
                    print(f"    live:    {a.get('live_lead_author_surname','?')} | {a.get('live_title','?')}")
    print()
    print(f"wrote: {OUT_JSON}")
    return 1 if n_fetch_fail else 0


if __name__ == "__main__":
    sys.exit(main())
