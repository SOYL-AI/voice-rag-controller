"""Build research-v2 dataset D1 (`datasets/v2/d1_open/`).

D1 is public open-domain QA where the need for retrieval varies continuously
with entity popularity:

* PopQA (Mallen et al., ACL 2023; MIT licence), stratified by log subject
  popularity (equal counts per quantile bin), restricted to items whose
  question stays unambiguous when spoken (the subject is the primary
  Wikipedia referent of its name and the question string is unique).
* HotpotQA distractor dev (Yang et al., EMNLP 2018; CC BY-SA 4.0), bridge and
  comparison questions with non-yes/no answers.

The retrieval corpus is the subject's Wikipedia article for every PopQA item
(chunked to <= 350-word passages), hard negatives (lead sections of
similarly named Wikipedia pages from Wikipedia search, and of other PopQA
subjects sharing the relation), and all ten HotpotQA context paragraphs of
every sampled HotpotQA item.

Everything is deterministic given the seed and the cached raw fetches.
Raw downloads are cached under `<out>/.cache/` (gitignored); a rebuild with
`--offline` uses the cache only and must reproduce the committed files
byte-for-byte.

Usage (from the repo root, any Python >= 3.9 with `requests`):

    set PYTHONUTF8=1
    python datasets/v2/build_d1_open.py                # fetch + build (5,000 PopQA items)
    python datasets/v2/build_d1_open.py --offline      # rebuild from cache
    python datasets/v2/build_d1_open.py --n-popqa 1000 --out <dir>   # smaller pilot build

Items also carry `meta.bin_rank` (rank of the item inside its popularity bin
under the seeded hash order), so "bin_rank < k" selects a nested pilot subset
of k items per bin from the full build.

No model is run by this script.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "d1_open"

DATASET = "d1_open"
DOMAIN = "open_domain"
SEED = 20261014

# Pinned upstream revisions.
POPQA_REV = "098765c79ea10a2cb19c828324e33281b8336ec0"
POPQA_URL = f"https://huggingface.co/datasets/akariasai/PopQA/resolve/{POPQA_REV}/test.tsv"
HOTPOT_ROWS_API = "https://datasets-server.huggingface.co/rows"
HOTPOT_HF_DATASET = "hotpotqa/hotpot_qa"
HOTPOT_HF_REV = "1908d6afbbead072334abe2965f91bd2709910ab"  # revision at build time (rows API is unpinned)
HOTPOT_N_ROWS = 7405
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "voice-rag-controller-d1-builder/1.0 "
    "(https://github.com/SOYL-AI/voice-rag-controller; research dataset construction)"
)

# --- PopQA filtering -------------------------------------------------------
# Spoken questions lose quotes, italics and capitalisation, so a subject is
# kept only if it is the primary Wikipedia referent of its name (its
# `s_wiki_title` has no parenthetical disambiguator such as "(film)") and its
# question string occurs once in PopQA (duplicated strings point at different
# entities). This item-level rule removes 63-76% of the creative-work
# relations (director, screenwriter, producer, composer, genre, author) and
# 3-14% of the others. Only `color` is dropped as a whole relation: 34 items
# whose answers are multi-valued and ill-defined (flags have several colours).
DROPPED_RELATIONS = {
    "color": "multi-valued, ill-defined answer (e.g. flag colours); 34 items",
}
# Relations whose gold answer is one of several valid values; flagged in
# item meta so closed-book scoring can treat them leniently or exclude them.
MULTI_VALUED_RELATIONS = {"genre", "occupation"}
N_POP_BINS = 5
# Leave headroom in each (bin, relation) cell for items dropped by the
# gold-passage check when computing quotas.
CELL_CAP_FRACTION = 0.7

# --- HotpotQA filtering ----------------------------------------------------
HOTPOT_MAX_Q_WORDS = 30
HOTPOT_TYPE_QUOTA = {"bridge": 150, "comparison": 150}

# --- Corpus / chunking -----------------------------------------------------
MAX_WORDS = 350
SECTION_BREAK_MIN_WORDS = 120  # break at a section boundary once a passage has this many words
MIN_PASSAGE_WORDS = 40  # shorter passages are folded into the previous one when it fits
GOLD_ARTICLE_MAX_PASSAGES = 20
NEG_LEAD_MAX_PASSAGES = 2
N_NAME_NEGATIVES = 3
N_NAME_SEARCH = 6
N_RELATION_NEGATIVES = 2
N_RELATION_CANDIDATES = 4
WIKI_LICENCE = "CC BY-SA 4.0 (Wikipedia text)"
HOTPOT_LICENCE = "CC BY-SA 4.0 (HotpotQA)"
POPQA_LICENCE = "MIT (PopQA)"
SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "notes and references",
    "references and notes", "works cited", "gallery", "external sources",
    "explanatory notes", "general references", "cited sources", "literature",
}
SPLIT_DEV_THRESHOLD = 300  # sha256(id) mod 1000 < 300 -> dev (CONTRACTS.md section 7)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def sha_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def order_key(*parts) -> str:
    return hashlib.sha256(("|".join(str(p) for p in (SEED,) + parts)).encode("utf-8")).hexdigest()


def split_of(item_id: str) -> str:
    return "dev" if sha_int(item_id) % 1000 < SPLIT_DEV_THRESHOLD else "test"


# --- text utils ------------------------------------------------------------
_CP1252_RUN = re.compile(
    "[\u0080-ÿŒœŠšŸŽžƒˆ˜"
    "–—‘’‚“”„†‡•…"
    "‰‹›€™]{2,}"
)


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 text that was decoded as cp1252 (e.g. 'â€“' -> '–').

    Only runs of characters that cp1252 can encode are re-decoded, and only
    when the bytes are valid UTF-8, so correct Latin-1 text is left alone.
    """
    def repl(m):
        run = m.group(0)
        try:
            return run.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return run
    return _CP1252_RUN.sub(repl, s)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = re.sub(r"[^\w]+", " ", s)
    return " " + " ".join(s.split()) + " "


def alias_in(alias: str, text: str, text_norm: str) -> bool:
    a = norm(alias)
    if a.strip() == "":
        return False
    if len(a.strip()) <= 3:
        # Short aliases ("US", "UK", "pol") must match case-sensitively as a word.
        return re.search(r"(?<!\w)" + re.escape(alias.strip()) + r"(?!\w)", text) is not None
    return a in text_norm


def n_words(s: str) -> int:
    return len(s.split())


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _units(paragraph: str):
    """Yield <= MAX_WORDS units from a paragraph (sentences, then word windows)."""
    if n_words(paragraph) <= MAX_WORDS:
        yield paragraph
        return
    buf = []
    for sent in _SENT_SPLIT.split(paragraph):
        words = sent.split()
        while len(words) > MAX_WORDS:
            if buf:
                yield " ".join(buf)
                buf = []
            yield " ".join(words[:MAX_WORDS])
            words = words[MAX_WORDS:]
        if len(buf) + len(words) > MAX_WORDS:
            yield " ".join(buf)
            buf = []
        buf.extend(words)
    if buf:
        yield " ".join(buf)


_HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")


def chunk_article(extract: str):
    """Split a plain-text Wikipedia extract into passages of <= MAX_WORDS words.

    Returns a list of {"section": str, "text": str}. Reference-type sections
    are skipped. Paragraphs are packed greedily; a new passage starts at a
    section boundary once the current one has SECTION_BREAK_MIN_WORDS words.
    """
    units = []  # (section, text, starts_section)
    section = "Introduction"
    skip_level = None
    new_section = True
    for line in extract.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _HEADING.match(line)
        if m:
            level, name = len(m.group(1)), m.group(2)
            if skip_level is not None and level > skip_level:
                continue
            skip_level = level if name.strip().casefold() in SKIP_SECTIONS else None
            section = name
            new_section = True
            continue
        if skip_level is not None:
            continue
        for u in _units(line):
            units.append((section, u, new_section))
            new_section = False

    passages = []
    cur, cur_n, cur_sec = [], 0, None
    for sec, text, starts in units:
        w = n_words(text)
        if cur and (cur_n + w > MAX_WORDS or (starts and cur_n >= SECTION_BREAK_MIN_WORDS)):
            passages.append({"section": cur_sec, "text": "\n".join(cur)})
            cur, cur_n = [], 0
        if not cur:
            cur_sec = sec
        cur.append(text)
        cur_n += w
    if cur:
        passages.append({"section": cur_sec, "text": "\n".join(cur)})
    # Fold short fragments (e.g. a tail section of one list line) into the
    # previous passage when that stays within MAX_WORDS.
    merged = []
    for p in passages:
        if (merged and n_words(p["text"]) < MIN_PASSAGE_WORDS
                and n_words(merged[-1]["text"]) + n_words(p["text"]) <= MAX_WORDS):
            merged[-1] = {"section": merged[-1]["section"],
                          "text": merged[-1]["text"] + "\n" + p["text"]}
        else:
            merged.append(p)
    return merged


# --- HTTP with on-disk cache -----------------------------------------------
class Fetcher:
    def __init__(self, cache_dir: Path, offline: bool, min_interval: float):
        self.cache_dir = cache_dir
        self.offline = offline
        self.min_interval = min_interval
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last = 0.0
        self.n_net = 0
        self.n_cache = 0

    def _path(self, ns: str, key: str) -> Path:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / ns / h[:2] / f"{h}.json"

    def get_json(self, ns: str, url: str, params: dict):
        key = url + "?" + json.dumps(params, sort_keys=True, ensure_ascii=False)
        p = self._path(ns, key)
        if p.exists():
            self.n_cache += 1
            return json.loads(p.read_text(encoding="utf-8"))
        if self.offline:
            raise RuntimeError(f"--offline and not cached: {key}")
        delay = 2.0
        min_interval = max(self.min_interval, 1.0) if ns == "hotpot_rows" else self.min_interval
        for attempt in range(8):
            wait = min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as e:
                log(f"  network error ({e}); retry in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                sl = float(ra) if ra and ra.isdigit() else delay
                log(f"  HTTP {r.status_code}; retry in {sl:.0f}s")
                time.sleep(sl)
                delay *= 2
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                log(f"  maxlag; retry in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            self.n_net += 1
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        raise RuntimeError(f"giving up on {key}")


class Wiki:
    def __init__(self, fetcher: Fetcher):
        self.f = fetcher
        self._articles = {}
        self._leads = {}

    def _q(self, params):
        base = {"format": "json", "formatversion": "2", "maxlag": "5"}
        base.update(params)
        return self.f.get_json("wiki", WIKI_API, base)

    def article(self, title: str):
        """Full plain-text article (following redirects) or None."""
        if title in self._articles:
            return self._articles[title]
        data = self._q({
            "action": "query", "prop": "extracts|info|pageprops", "ppprop": "disambiguation",
            "explaintext": "1", "exsectionformat": "wiki", "redirects": "1", "titles": title,
        })
        pages = data.get("query", {}).get("pages", [])
        out = None
        if pages:
            pg = pages[0]
            if not pg.get("missing") and not pg.get("invalid") and pg.get("extract"):
                out = {
                    "pageid": pg["pageid"],
                    "title": pg["title"],
                    "revid": pg.get("lastrevid"),
                    "disambiguation": "disambiguation" in (pg.get("pageprops") or {}),
                    "extract": pg["extract"],
                }
        self._articles[title] = out
        return out

    @staticmethod
    def _page(pg):
        if not pg or pg.get("missing") or pg.get("invalid") or not pg.get("extract", "").strip():
            return None
        return {
            "pageid": pg["pageid"], "title": pg["title"], "revid": pg.get("lastrevid"),
            "disambiguation": "disambiguation" in (pg.get("pageprops") or {}),
            "extract": pg["extract"], "index": pg.get("index"),
        }

    def _query_all(self, params):
        """Run a query, following `continue`, and merge the returned pages."""
        pages, extra, cont = {}, {"normalized": [], "redirects": []}, {}
        for _ in range(50):
            data = self._q({**params, **cont})
            q = data.get("query", {})
            for k in extra:
                extra[k].extend(q.get(k, []))
            for pg in q.get("pages", []):
                key = pg.get("pageid", pg.get("title"))
                if key in pages:
                    pages[key].update({k: v for k, v in pg.items() if v})
                else:
                    pages[key] = dict(pg)
            if "continue" not in data:
                break
            cont = data["continue"]
        return list(pages.values()), extra

    LEAD_PARAMS = {"action": "query", "prop": "extracts|info|pageprops", "ppprop": "disambiguation",
                   "exintro": "1", "explaintext": "1", "exlimit": "20"}

    def leads(self, titles):
        """Lead-section extracts for many titles, 20 per request (follows redirects)."""
        todo = sorted({t for t in titles if t not in self._leads and "|" not in t})
        for i in range(0, len(todo), 20):
            batch = todo[i:i + 20]
            pages, extra = self._query_all({**self.LEAD_PARAMS, "redirects": "1",
                                            "titles": "|".join(batch)})
            by_title = {pg.get("title"): pg for pg in pages}
            for t in batch:
                cur = t
                for k in ("normalized", "redirects"):
                    for m in extra[k]:
                        if m["from"] == cur:
                            cur = m["to"]
                self._leads[t] = self._page(by_title.get(cur))
            if (i // 20) % 50 == 0:
                log(f"  lead batches {i // 20 + 1}/{(len(todo) + 19) // 20} "
                    f"(net={self.f.n_net} cache={self.f.n_cache})")
        return {t: self._leads.get(t) for t in titles}

    def search_leads(self, query: str, limit: int = N_NAME_SEARCH):
        """Wikipedia search results with their lead extracts, in rank order (one request)."""
        pages, _ = self._query_all({**self.LEAD_PARAMS, "generator": "search", "gsrsearch": query,
                                    "gsrlimit": str(limit), "gsrnamespace": "0"})
        out = [self._page(pg) for pg in pages]
        return sorted((p for p in out if p), key=lambda p: (p["index"] or 0, p["pageid"]))


# --- PopQA -----------------------------------------------------------------
def load_popqa(fetcher: Fetcher):
    path = fetcher.cache_dir / "raw" / "popqa_test.tsv"
    if not path.exists():
        if fetcher.offline:
            raise RuntimeError(f"--offline and missing {path}")
        log(f"downloading PopQA from {POPQA_URL}")
        r = fetcher.session.get(POPQA_URL, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        r["s_pop"] = int(r["s_pop"])
        r["o_pop"] = int(r["o_pop"])
        r["possible_answers"] = json.loads(r["possible_answers"])
    return rows


def popqa_pool(rows, report):
    """Apply relation- and item-level filters; return the eligible pool."""
    q_count = collections.Counter(r["question"] for r in rows)
    pool, drops = [], collections.Counter()
    for r in rows:
        if r["prop"] in DROPPED_RELATIONS:
            drops[f"relation:{r['prop']}"] += 1
        elif q_count[r["question"]] > 1:
            drops["duplicate_question_text"] += 1
        elif "(" in r["s_wiki_title"]:
            drops["subject_needs_disambiguator"] += 1
        elif not [a for a in r["possible_answers"] if a.strip()]:
            drops["no_answers"] += 1
        else:
            pool.append(r)
    report["popqa_source_rows"] = len(rows)
    report["popqa_pool_filter_drops"] = dict(sorted(drops.items()))
    report["popqa_eligible_pool"] = len(pool)
    return pool


def pop_bin_edges(pool):
    vals = sorted(math.log10(r["s_pop"]) for r in pool)
    n = len(vals)
    return [vals[int(n * k / N_POP_BINS)] for k in range(1, N_POP_BINS)]


def pop_bin(s_pop, edges):
    v = math.log10(s_pop)
    return sum(1 for e in edges if v >= e)


def waterfill(total, caps):
    """Spread `total` as evenly as possible over keys with per-key caps."""
    alloc = {k: 0 for k in caps}
    remaining = total
    open_keys = sorted(k for k in caps if caps[k] > 0)
    while remaining > 0 and open_keys:
        share = max(1, remaining // len(open_keys))
        for k in list(open_keys):
            if remaining == 0:
                break
            add = min(share, caps[k] - alloc[k], remaining)
            alloc[k] += add
            remaining -= add
            if alloc[k] >= caps[k]:
                open_keys.remove(k)
    return alloc


def check_gold(row, wiki: Wiki):
    """Fetch the subject article; return (article, passages, gold_idx) or a drop reason."""
    art = wiki.article(row["s_wiki_title"])
    if art is None:
        return "wiki_article_missing"
    if art["disambiguation"]:
        return "wiki_article_is_disambiguation_page"
    passages = chunk_article(art["extract"])[:GOLD_ARTICLE_MAX_PASSAGES]
    if not passages:
        return "wiki_article_empty_after_chunking"
    aliases = [a for a in row["possible_answers"] if a.strip()]
    gold = []
    for i, p in enumerate(passages):
        tn = norm(p["text"])
        if any(alias_in(a, p["text"], tn) for a in aliases):
            gold.append(i)
    if not gold:
        return "no_alias_in_subject_article"
    return art, passages, gold


def sample_popqa(pool, n_target, wiki: Wiki, report):
    edges = pop_bin_edges(pool)
    report["popqa_pop_bin_edges_s_pop"] = [round(10 ** e, 1) for e in edges]
    cells = collections.defaultdict(list)
    for r in pool:
        r["_bin"] = pop_bin(r["s_pop"], edges)
        cells[(r["_bin"], r["prop"])].append(r)
    for k in cells:
        cells[k].sort(key=lambda r: order_key("popqa", r["id"]))

    relations = sorted({r["prop"] for r in pool})
    per_bin = [n_target // N_POP_BINS + (1 if b < n_target % N_POP_BINS else 0) for b in range(N_POP_BINS)]
    accepted, drops = [], collections.Counter()
    drop_by_bin = collections.Counter()
    quotas_report = {}
    attempted = 0
    for b in range(N_POP_BINS):
        caps = {p: int(CELL_CAP_FRACTION * len(cells.get((b, p), []))) for p in relations}
        quota = waterfill(per_bin[b], caps)
        quotas_report[b] = {p: q for p, q in quota.items() if q}
        cursor = {p: 0 for p in relations}
        got = {p: 0 for p in relations}

        def try_next(p):
            nonlocal attempted
            cands = cells.get((b, p), [])
            while cursor[p] < len(cands):
                r = cands[cursor[p]]
                cursor[p] += 1
                attempted += 1
                res = check_gold(r, wiki)
                if isinstance(res, str):
                    drops[res] += 1
                    drop_by_bin[b] += 1
                    continue
                art, passages, gold = res
                r["_article"], r["_passages"], r["_gold"] = art, passages, gold
                accepted.append(r)
                got[p] += 1
                if attempted % 50 == 0:
                    log(f"  popqa attempted={attempted} accepted={len(accepted)} "
                        f"(net={wiki.f.n_net} cache={wiki.f.n_cache})")
                return True
            return False

        # Phase 1: per-relation quotas.
        for p in relations:
            while got[p] < quota[p] and try_next(p):
                pass
        # Phase 2: fill any shortfall in this bin round-robin over relations.
        while sum(got.values()) < per_bin[b]:
            progressed = False
            for p in relations:
                if sum(got.values()) >= per_bin[b]:
                    break
                progressed |= try_next(p)
            if not progressed:
                break
        log(f"bin {b}: accepted {sum(got.values())}/{per_bin[b]}")
    report["popqa_target"] = n_target
    report["popqa_quota_by_bin"] = {str(k): v for k, v in quotas_report.items()}
    report["popqa_attempted"] = attempted
    report["popqa_gold_check_drops"] = dict(sorted(drops.items()))
    report["popqa_gold_check_drops_by_bin"] = {str(k): v for k, v in sorted(drop_by_bin.items())}
    return accepted, edges


# --- HotpotQA --------------------------------------------------------------
def load_hotpot(fetcher: Fetcher, hotpot_json: Path | None):
    """Return HotpotQA distractor dev rows in the official JSON layout."""
    if hotpot_json is not None:
        return json.loads(Path(hotpot_json).read_text(encoding="utf-8"))
    rows = []
    for off in range(0, HOTPOT_N_ROWS, 100):
        data = fetcher.get_json("hotpot_rows", HOTPOT_ROWS_API, {
            "dataset": HOTPOT_HF_DATASET, "config": "distractor", "split": "validation",
            "offset": str(off), "length": "100",
        })
        for wrapped in data["rows"]:
            r = wrapped["row"]
            rows.append({
                "_id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "type": r["type"],
                "level": r["level"],
                "supporting_facts": [[t, s] for t, s in zip(r["supporting_facts"]["title"],
                                                           r["supporting_facts"]["sent_id"])],
                "context": [[t, s] for t, s in zip(r["context"]["title"], r["context"]["sentences"])],
            })
    if len(rows) != HOTPOT_N_ROWS:
        raise RuntimeError(f"expected {HOTPOT_N_ROWS} HotpotQA dev rows, got {len(rows)}")
    return rows


def pack_sentences(sents):
    """Pack a HotpotQA paragraph's sentences into <= MAX_WORDS-word parts.

    Returns [(text, {sentence indices})]. Almost all paragraphs give one part.
    """
    parts, cur, cur_ids, cur_n = [], [], set(), 0
    for i, s in enumerate(sents):
        w = n_words(s)
        if cur and cur_n + w > MAX_WORDS:
            parts.append(("".join(cur).strip(), cur_ids))
            cur, cur_ids, cur_n = [], set(), 0
        if w > MAX_WORDS:  # a single over-long sentence: hard word windows
            words = s.split()
            for k in range(0, len(words), MAX_WORDS):
                parts.append((" ".join(words[k:k + MAX_WORDS]), {i}))
            continue
        cur.append(s)
        cur_ids.add(i)
        cur_n += w
    if cur:
        parts.append(("".join(cur).strip(), cur_ids))
    return [(t, ids) for t, ids in parts if t]


def hotpot_doc_id(title: str, text: str) -> str:
    return "d1-hp-" + hashlib.sha1((title + "\n" + text).encode("utf-8")).hexdigest()[:16]


def sample_hotpot(rows, report):
    drops = collections.Counter()
    by_type = collections.defaultdict(list)
    for r in rows:
        q = fix_mojibake(r["question"]).strip()
        a = fix_mojibake(r["answer"]).strip()
        if a.lower() in ("yes", "no"):
            drops["yes_no_answer"] += 1
            continue
        if n_words(q) > HOTPOT_MAX_Q_WORDS:
            drops[f"question_over_{HOTPOT_MAX_Q_WORDS}_words"] += 1
            continue
        by_type[r["type"]].append((r, q, a))
    report["hotpot_source_rows"] = len(rows)
    report["hotpot_pool_filter_drops"] = dict(sorted(drops.items()))
    report["hotpot_eligible_pool"] = {t: len(v) for t, v in sorted(by_type.items())}

    accepted, gdrops = [], collections.Counter()
    for t, quota in HOTPOT_TYPE_QUOTA.items():
        cands = sorted(by_type[t], key=lambda x: order_key("hotpot", x[0]["_id"]))
        got = 0
        for r, q, a in cands:
            if got >= quota:
                break
            sup_sents = collections.defaultdict(set)
            for sf_title, sf_sent in r["supporting_facts"]:
                sup_sents[fix_mojibake(sf_title)].add(int(sf_sent))
            sup_titles = sorted(sup_sents)
            paras, gold = [], []
            for title, sents in r["context"]:
                title = fix_mojibake(title)
                for text, sent_ids in pack_sentences([fix_mojibake(s) for s in sents]):
                    paras.append((title, text))
                    if title in sup_sents and (sent_ids & sup_sents[title]
                                               or max(sup_sents[title]) >= len(sents)):
                        gold.append((title, text))
            if len({ti for ti, _ in gold}) != len(sup_titles):
                gdrops["supporting_paragraph_missing"] += 1
                continue
            if not any(alias_in(a, ti + "\n" + tx, norm(ti + " " + tx)) for ti, tx in gold):
                gdrops["answer_not_in_gold_paragraphs"] += 1
                continue
            accepted.append({"row": r, "question": q, "answer": a, "paras": paras,
                             "gold": gold, "sup_titles": sup_titles})
            got += 1
        log(f"hotpot {t}: accepted {got}/{quota}")
    report["hotpot_gold_check_drops"] = dict(sorted(gdrops.items()))
    return accepted


# --- negatives -------------------------------------------------------------
def usable_negative(page, gold_pageid: int) -> bool:
    return (page is not None and not page["disambiguation"] and page["pageid"] != gold_pageid
            and "(disambiguation)" not in page["title"] and not page["title"].startswith("List of"))


def relation_negative_candidates(row, by_prop_obj, by_prop):
    same_obj = [r for r in by_prop_obj[(row["prop"], row["obj"])] if r["s_wiki_title"] != row["s_wiki_title"]]
    same_prop = [r for r in by_prop[row["prop"]]
                 if r["s_wiki_title"] != row["s_wiki_title"] and r["obj"] != row["obj"]]
    same_obj.sort(key=lambda r: order_key("relneg", row["id"], r["id"]))
    same_prop.sort(key=lambda r: order_key("relneg", row["id"], r["id"]))
    return [r["s_wiki_title"] for r in same_obj + same_prop]


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution (regularised gamma series)."""
    if x <= 0:
        return 1.0
    a, z = df / 2.0, x / 2.0
    term = total = 1.0 / a
    n = 0
    while term > 1e-15 * total and n < 10000:
        n += 1
        term *= z / (a + n)
        total += term
    lower = math.exp(-z + a * math.log(z) - math.lgamma(a)) * total
    return max(0.0, 1.0 - lower)


def check_and_balance_splits(items, stratum):
    """Report dev/test balance per stratum; rebalance only badly unbalanced strata."""
    p = SPLIT_DEV_THRESHOLD / 1000.0

    def table(its):
        t = collections.defaultdict(lambda: {"dev": 0, "test": 0})
        for it in its:
            t[stratum(it)][it["split"]] += 1
        return t

    def family(k):
        return k.rsplit("_", 1)[0] if k.startswith("multihop") else "popqa"

    def homogeneity(t, keys):
        dev = sum(t[k]["dev"] for k in keys)
        n = sum(t[k]["dev"] + t[k]["test"] for k in keys)
        x2 = 0.0
        for k in keys:
            nk = t[k]["dev"] + t[k]["test"]
            for col, tot in (("dev", dev), ("test", n - dev)):
                e = nk * tot / n
                x2 += (t[k][col] - e) ** 2 / e if e else 0.0
        return x2, len(keys) - 1

    t = table(items)
    families = collections.defaultdict(list)
    for k in sorted(t):
        families[family(k)].append(k)
    bad = set()
    fam_report = {}
    for fam, keys in families.items():
        x2, df = homogeneity(t, keys)
        pval = chi2_sf(x2, df) if df > 0 else 1.0
        fam_report[fam] = {"chi2": round(x2, 3), "df": df, "p": round(pval, 4)}
        for k in keys:
            nk = t[k]["dev"] + t[k]["test"]
            z = (t[k]["dev"] / nk - p) / math.sqrt(p * (1 - p) / nk)
            if abs(z) > 2 or pval < 0.05:
                bad.add(k)
    adjusted = sorted(bad)
    for k in adjusted:
        members = sorted((it for it in items if stratum(it) == k),
                         key=lambda it: (sha_int(it["id"]) % 1000, it["id"]))
        n_dev = round(p * len(members))
        for i, it in enumerate(members):
            it["split"] = "dev" if i < n_dev else "test"
    t = table(items)
    report = {}
    for k in sorted(t):
        d, te = t[k]["dev"], t[k]["test"]
        z = (d / (d + te) - p) / math.sqrt(p * (1 - p) / (d + te))
        report[k] = {"dev": d, "test": te, "dev_frac": round(d / (d + te), 3), "z": round(z, 2),
                     "rebalanced": k in bad}
    report["homogeneity_before_adjustment"] = fam_report
    return report, adjusted


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=None, help="default: <out>/.cache")
    ap.add_argument("--offline", action="store_true", help="use cached raw fetches only")
    ap.add_argument("--min-interval", type=float, default=0.2,
                    help="minimum seconds between network requests (serial)")
    ap.add_argument("--n-popqa", type=int, default=5000)
    ap.add_argument("--hotpot-json", type=Path, default=None,
                    help="optional local hotpot_dev_distractor_v1.json instead of the HF rows API")
    args = ap.parse_args()

    out = args.out
    cache = args.cache or (out / ".cache")
    fetcher = Fetcher(cache, args.offline, args.min_interval)
    wiki = Wiki(fetcher)
    report = {"seed": SEED}

    # 1. PopQA
    rows = load_popqa(fetcher)
    pool = popqa_pool(rows, report)
    log(f"PopQA: {len(rows)} rows, eligible pool {len(pool)}")
    popqa, edges = sample_popqa(pool, args.n_popqa, wiki, report)

    # 2. HotpotQA
    hp_rows = load_hotpot(fetcher, args.hotpot_json)
    hotpot = sample_hotpot(hp_rows, report)

    # 3. Corpus: gold articles, negatives, HotpotQA paragraphs.
    corpus = {}
    page_docs = collections.defaultdict(list)

    def add_article(art, passages, max_passages, role):
        ids = []
        for i, p in enumerate(passages[:max_passages]):
            doc_id = f"d1-wiki-{art['pageid']}-{i:03d}"
            if doc_id not in corpus:
                corpus[doc_id] = {
                    "doc_id": doc_id, "dataset": DATASET, "domain": DOMAIN, "org": None,
                    "title": art["title"], "text": p["text"], "source": "wikipedia",
                    "licence": WIKI_LICENCE,
                    "meta": {"pageid": art["pageid"], "revid": art["revid"], "section": p["section"],
                             "passage_index": i, "roles": []},
                }
                page_docs[art["pageid"]].append(doc_id)
            if role not in corpus[doc_id]["meta"]["roles"]:
                corpus[doc_id]["meta"]["roles"].append(role)
            ids.append(doc_id)
        return ids

    by_bin = collections.defaultdict(list)
    for r in popqa:
        by_bin[r["_bin"]].append(r)
    for members in by_bin.values():
        members.sort(key=lambda r: order_key("popqa", r["id"]))
        for rank, r in enumerate(members):
            r["_bin_rank"] = rank

    gold_pageids = set()
    for r in popqa:
        ids = add_article(r["_article"], r["_passages"], GOLD_ARTICLE_MAX_PASSAGES, "popqa_subject")
        r["_gold_ids"] = [ids[i] for i in r["_gold"]]
        gold_pageids.add(r["_article"]["pageid"])

    by_prop_obj, by_prop = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        if r["prop"] not in DROPPED_RELATIONS:
            by_prop_obj[(r["prop"], r["obj"])].append(r)
            by_prop[r["prop"]].append(r)

    def add_negative(page, role):
        if page["pageid"] in page_docs:  # already present (e.g. another item's gold article)
            for doc_id in page_docs[page["pageid"]]:
                if role not in corpus[doc_id]["meta"]["roles"]:
                    corpus[doc_id]["meta"]["roles"].append(role)
            return
        add_article(page, chunk_article(page["extract"]), NEG_LEAD_MAX_PASSAGES, role)

    neg_stats = collections.Counter()
    # Name negatives: lead sections of the top Wikipedia search hits for the subject name.
    for n, r in enumerate(popqa, 1):
        gp = r["_article"]["pageid"]
        got = 0
        for page in wiki.search_leads(r["subj"]):
            if got >= N_NAME_NEGATIVES:
                break
            if not usable_negative(page, gp):
                neg_stats["name_negative_unusable"] += 1
                continue
            add_negative(page, "name_negative")
            neg_stats["name_negative_pages"] += 1
            got += 1
        if n % 250 == 0:
            log(f"  name negatives for {n}/{len(popqa)} items (corpus={len(corpus)}, "
                f"net={fetcher.n_net} cache={fetcher.n_cache})")
    # Relation negatives: lead sections of other PopQA subjects with the same
    # relation (same object first), fetched 20 titles per request.
    rel_cands = {r["id"]: relation_negative_candidates(r, by_prop_obj, by_prop)[:N_RELATION_CANDIDATES]
                 for r in popqa}
    leads = wiki.leads([t for c in rel_cands.values() for t in c])
    for r in popqa:
        gp = r["_article"]["pageid"]
        got = 0
        for title in rel_cands[r["id"]]:
            if got >= N_RELATION_NEGATIVES:
                break
            page = leads.get(title)
            if not usable_negative(page, gp):
                neg_stats["relation_negative_unusable"] += 1
                continue
            add_negative(page, "relation_negative")
            neg_stats["relation_negative_pages"] += 1
            got += 1
    report["negative_fetch_stats"] = dict(sorted(neg_stats.items()))

    for h in hotpot:
        gold_set = set(h["gold"])
        for title, text in h["paras"]:
            doc_id = hotpot_doc_id(title, text)
            if doc_id not in corpus:
                corpus[doc_id] = {
                    "doc_id": doc_id, "dataset": DATASET, "domain": DOMAIN, "org": None,
                    "title": title, "text": text, "source": "hotpotqa",
                    "licence": HOTPOT_LICENCE, "meta": {"roles": []},
                }
            role = "hotpot_supporting" if (title, text) in gold_set else "hotpot_distractor"
            if role not in corpus[doc_id]["meta"]["roles"]:
                corpus[doc_id]["meta"]["roles"].append(role)
        h["gold_ids"] = sorted({hotpot_doc_id(ti, tx) for ti, tx in h["gold"]})

    for d in corpus.values():
        d["meta"]["roles"].sort()

    # 4. Items
    items = []
    for r in popqa:
        iid = f"{DATASET}-popqa-{r['id']}"
        items.append({
            "id": iid, "dataset": DATASET, "domain": DOMAIN, "org": None,
            "question": r["question"], "expected_answer": r["obj"],
            "answer_aliases": [a for a in r["possible_answers"] if a.strip()],
            "gold_doc_ids": r["_gold_ids"], "item_type": "popqa",
            "retrieval_needed_authored": None, "pair_id": None,
            "popularity": r["s_pop"], "history": [], "split": split_of(iid),
            "meta": {
                "source": "PopQA", "source_id": r["id"], "licence": POPQA_LICENCE,
                "relation": r["prop"], "subject": r["subj"], "s_wiki_title": r["s_wiki_title"],
                "resolved_wiki_title": r["_article"]["title"], "pageid": r["_article"]["pageid"],
                "revid": r["_article"]["revid"], "s_pop": r["s_pop"], "o_pop": r["o_pop"],
                "log10_pop": round(math.log10(r["s_pop"]), 4), "pop_bin": r["_bin"],
                # gold_doc_ids = every subject-article passage containing an alias;
                # primary = the first of them in article order (usually the lead).
                "primary_gold_doc_id": r["_gold_ids"][0],
                "bin_rank": r["_bin_rank"],
                "multi_valued_answer": r["prop"] in MULTI_VALUED_RELATIONS,
                "subject_wikidata": r["s_uri"], "object_wikidata": r["o_uri"],
            },
        })
    for h in hotpot:
        r = h["row"]
        iid = f"{DATASET}-hp-{r['_id']}"
        items.append({
            "id": iid, "dataset": DATASET, "domain": DOMAIN, "org": None,
            "question": h["question"], "expected_answer": h["answer"],
            "answer_aliases": [h["answer"]], "gold_doc_ids": h["gold_ids"],
            "item_type": "multihop", "retrieval_needed_authored": None, "pair_id": None,
            "popularity": None, "history": [], "split": split_of(iid),
            "meta": {
                "source": "HotpotQA distractor dev", "source_id": r["_id"],
                "licence": HOTPOT_LICENCE, "hotpot_type": r["type"], "level": r["level"],
                "supporting_titles": h["sup_titles"],
            },
        })
    items.sort(key=lambda x: x["id"])

    # 5. Splits + stratification check
    def stratum(it):
        if it["item_type"] == "popqa":
            return f"popqa_bin{it['meta']['pop_bin']}"
        return f"multihop_{it['meta']['hotpot_type']}"

    strat_report, adjusted = check_and_balance_splits(items, stratum)
    splits = {
        "dev": [it["id"] for it in items if it["split"] == "dev"],
        "test": [it["id"] for it in items if it["split"] == "test"],
        "seed": SEED,
        "method": ("sha256(id) mod 1000 < 300 -> dev, else test (CONTRACTS.md section 7). "
                   "A stratum is rebalanced only if badly unbalanced (|z| > 2 against 0.30, or "
                   "chi-square homogeneity p < 0.05 within its family); rebalancing assigns the "
                   "round(0.3 n) items with the smallest sha256(id) mod 1000 to dev. "
                   + ("Rebalanced strata: " + ", ".join(adjusted) if adjusted
                      else "No stratum met the criterion; no adjustment was made.")),
        "stratification": strat_report,
    }

    # 6. Write
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for doc_id in sorted(corpus):
            fh.write(json.dumps(corpus[doc_id], ensure_ascii=False) + "\n")
    (out / "splits.json").write_text(json.dumps(splits, indent=1, ensure_ascii=False) + "\n",
                                     encoding="utf-8", newline="\n")

    stats = build_stats(items, corpus, report)
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n",
                                    encoding="utf-8", newline="\n")
    log(json.dumps({"items": len(items), "corpus": len(corpus),
                    "dev": len(splits["dev"]), "test": len(splits["test"])}))
    log(f"network requests: {fetcher.n_net}, cache hits: {fetcher.n_cache}")


def build_stats(items, corpus, report):
    def dist(vals):
        vals = sorted(vals)
        n = len(vals)
        return {"n": n, "mean": round(sum(vals) / n, 2), "min": vals[0], "p50": vals[n // 2],
                "p90": vals[int(n * 0.9)], "max": vals[-1]} if n else {}

    s = {"build_report": report}
    c = collections.Counter()
    for it in items:
        c[(it["item_type"], it["split"])] += 1
    s["items_by_type_split"] = {f"{t}/{sp}": n for (t, sp), n in sorted(c.items())}
    pb = collections.Counter((it["meta"]["pop_bin"], it["split"]) for it in items if it["item_type"] == "popqa")
    s["popqa_by_pop_bin_split"] = {f"bin{b}/{sp}": n for (b, sp), n in sorted(pb.items())}
    rel = collections.Counter((it["meta"]["relation"], it["meta"]["pop_bin"]) for it in items
                              if it["item_type"] == "popqa")
    rels = sorted({r for r, _ in rel})
    s["popqa_relation_by_pop_bin"] = {r: [rel[(r, b)] for b in range(N_POP_BINS)] for r in rels}
    s["popqa_pop_by_bin"] = {
        f"bin{b}": dist([it["popularity"] for it in items
                         if it["item_type"] == "popqa" and it["meta"]["pop_bin"] == b])
        for b in range(N_POP_BINS)}
    hp = collections.Counter((it["meta"]["hotpot_type"], it["split"]) for it in items
                             if it["item_type"] == "multihop")
    s["multihop_by_type_split"] = {f"{t}/{sp}": n for (t, sp), n in sorted(hp.items())}
    for t in ("popqa", "multihop"):
        s[f"question_words_{t}"] = dist([n_words(it["question"]) for it in items if it["item_type"] == t])
        s[f"gold_docs_per_item_{t}"] = dist([len(it["gold_doc_ids"]) for it in items if it["item_type"] == t])
    src = collections.Counter(d["source"] for d in corpus.values())
    s["corpus_passages_by_source"] = dict(sorted(src.items()))
    roles = collections.Counter(r for d in corpus.values() for r in d["meta"]["roles"])
    s["corpus_passages_by_role"] = dict(sorted(roles.items()))
    s["corpus_wikipedia_articles"] = len({d["meta"]["pageid"] for d in corpus.values() if d["source"] == "wikipedia"})
    s["corpus_passage_words"] = dist([n_words(d["text"]) for d in corpus.values()])
    s["corpus_total"] = len(corpus)
    return s


if __name__ == "__main__":
    main()
