"""Vendor the course's assignment pages into specs/ so work can proceed offline.

The cloud routine that maintains this repo cannot reach insper.github.io through its
egress proxy, so it reads these copies instead. Refresh from a machine that can:

    python specs/fetch_specs.py

Each page's <article> element is saved verbatim as HTML (no lossy markdown conversion),
plus an index.json with the fetch time, so a reader can tell how stale a copy is.
"""
import json, re, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = {
    # assignment specs for this semester
    "insper-2026.2/index.html": "https://insper.github.io/ann-dl/2026.2/",
    "insper-2026.2/exercises/data.html": "https://insper.github.io/ann-dl/2026.2/exercises/data/",
    "insper-2026.2/exercises/perceptron.html": "https://insper.github.io/ann-dl/2026.2/exercises/perceptron/",
    "insper-2026.2/exercises/mlp.html": "https://insper.github.io/ann-dl/2026.2/exercises/mlp/",
    "insper-2026.2/exercises/vae.html": "https://insper.github.io/ann-dl/2026.2/exercises/vae/",
    "insper-2026.2/exercises/submission.html": "https://insper.github.io/ann-dl/2026.2/exercises/submission/",
    # last semester's project pages: structural reference until 2026.2 publishes its own
    "insper-2025.2/projects/eda.html": "https://insper.github.io/ann-dl/2025.2/projects/eda/",
    "insper-2025.2/projects/classification.html": "https://insper.github.io/ann-dl/2025.2/projects/classification/",
    "insper-2025.2/projects/regression.html": "https://insper.github.io/ann-dl/2025.2/projects/regression/",
    "insper-2025.2/projects/generative.html": "https://insper.github.io/ann-dl/2025.2/projects/generative/",
    # repo structure template
    "template/index.html": "https://repo-classes.github.io/ann-dl/",
    "template/template.html": "https://repo-classes.github.io/ann-dl/template/",
    "template/exercises/data.html": "https://repo-classes.github.io/ann-dl/exercises/data/",
    "template/projects/eda.html": "https://repo-classes.github.io/ann-dl/projects/eda/",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "specs-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def article(html):
    m = re.search(r"<article[^>]*>(.*?)</article>", html, re.S)
    return m.group(1) if m else html


index = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pages": {}}
for rel, url in PAGES.items():
    out = HERE / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = article(fetch(url))
        out.write_text(f"<!-- source: {url} -->\n{body}", encoding="utf-8")
        index["pages"][rel] = {"url": url, "status": "ok", "bytes": len(body)}
        print(f"ok    {rel}")
    except Exception as exc:  # 404 = not published yet; keep going
        index["pages"][rel] = {"url": url, "status": str(exc)}
        if out.exists():
            out.unlink()
        print(f"skip  {rel}: {exc}", file=sys.stderr)

(HERE / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
