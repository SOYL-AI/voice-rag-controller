#!/usr/bin/env python3
"""Build the D2 ``d2_support`` benchmark from per-organisation raw files.

Input layout (one directory per fictional organisation)::

    datasets/v2/d2_support/raw/<domain>/<org_slug>/docs.jsonl    # corpus schema, CONTRACTS §3
    datasets/v2/d2_support/raw/<domain>/<org_slug>/items.jsonl   # item schema, CONTRACTS §2, local ids

Outputs (in ``datasets/v2/d2_support/``)::

    items.jsonl                 # global ids, global pair ids, split, enriched meta
    corpus.jsonl                # all documents
    splits.json                 # {"dev": [...], "test": [...], "seed": ..., "method": ...}
    stats.json                  # counts per domain / org / type / split, length stats
    validation_report.txt       # every ERROR and WARNING from the validator
    verification_sample.csv     # stratified 20% sample for human review

The builder works on whatever ``raw/<domain>/<org>/`` directories are present.
Global ids are stable under the addition of new domains or organisations
because they are derived from a fixed domain order and the org's position
within its own domain (see ``assign_ids``).

Usage::

    PYTHONUTF8=1 python datasets/v2/build_d2_support.py                 # validate + build
    PYTHONUTF8=1 python datasets/v2/build_d2_support.py --check-only    # validate only
    PYTHONUTF8=1 python datasets/v2/build_d2_support.py --check-only --domains telecom banking

Standard library only (no numpy / pyarrow), so it runs on any Python >= 3.9.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DATASET = "d2_support"
HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / DATASET / "raw"
DEFAULT_OUT = HERE / DATASET

# Fixed domain order -> stable global ids. Unknown domains are appended in
# alphabetical order after these, so adding a domain never renumbers another.
DOMAIN_ORDER = [
    "telecom",
    "banking",
    "insurance",
    "ecommerce",
    "airline",
    "healthcare",
    "utilities",
    "saas",
    "food_delivery",
    "education",
    "government",
    "automotive",
]

# Human-readable domain phrases for the system prompt template.
DOMAIN_LABELS = {
    "telecom": "telecom (mobile and broadband)",
    "banking": "banking and fintech",
    "insurance": "insurance",
    "ecommerce": "e-commerce and retail",
    "airline": "airline and travel",
    "healthcare": "healthcare (clinic and hospital)",
    "utilities": "utilities",
    "saas": "software and SaaS",
    "food_delivery": "food delivery",
    "education": "education and university admissions",
    "government": "government and citizen services",
    "automotive": "automotive service and dealership",
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a customer support voice agent for {org}, a {domain} company. "
    "Answer the caller briefly."
)

ITEM_TYPES_RETRIEVAL = ("org_specific", "counter_prior", "ambiguous")
ITEM_TYPES = ITEM_TYPES_RETRIEVAL + ("general",)
# Stratum of a pair group = its rarest retrieval-needed type.
TYPE_PRECEDENCE = ("ambiguous", "counter_prior", "org_specific")

ITEM_KEYS = {
    "id", "dataset", "domain", "org", "question", "expected_answer",
    "answer_aliases", "gold_doc_ids", "item_type", "retrieval_needed_authored",
    "pair_id", "popularity", "history", "meta",
}
DOC_KEYS = {"doc_id", "dataset", "domain", "org", "title", "text", "source", "licence"}

# Authoring targets from the D2 spec (deviations are warnings, not errors).
TARGET_ITEMS_PER_ORG = 30
TARGET_TYPE_FRACTIONS = {"org_specific": 0.35, "counter_prior": 0.20, "ambiguous": 0.10, "general": 0.35}
TYPE_FRACTION_TOLERANCE = 0.08
DOC_COUNT_RANGE = (12, 15)
DOC_WORD_RANGE = (120, 350)
HISTORY_FRACTION_RANGE = (0.10, 0.15)
PAIR_LENGTH_TOLERANCE = 3
QUESTION_WORD_RANGE = (4, 40)

DEV_FRACTION = 0.30
SPLIT_SALT = ""  # CONTRACTS §7 uses unsalted sha256

DEDUP_ERROR_RATIO = 0.92
DEDUP_WARN_RATIO = 0.82

# Generic words that may legitimately appear in a caller's question even when
# they are part of an organisation's name ("Kestrel Mobile" -> "mobile").
GENERIC_NAME_WORDS = {
    "the", "and", "of", "for", "a", "an", "&", "co", "company", "group", "ltd",
    "limited", "inc", "plc", "llc", "pvt", "private", "corp", "corporation",
    "mobile", "mobiles", "telecom", "telecoms", "networks", "network", "net",
    "broadband", "fibre", "fiber", "wireless", "connect", "communications",
    "bank", "banking", "finance", "financial", "fintech", "pay", "money",
    "capital", "credit", "savings", "trust", "wallet", "insurance", "assurance",
    "general", "health", "life", "mutual", "cover", "shield", "care", "secure",
    "retail", "store", "stores", "shop", "mart", "market", "home", "homes",
    "online", "direct", "express", "airways", "airlines", "air", "airline",
    "aviation", "travel", "travels", "tours", "holidays", "hospital",
    "hospitals", "clinic", "clinics", "medical", "healthcare", "diagnostics",
    "labs", "lab", "centre", "center", "services", "service", "solutions",
    "global", "international", "india", "national", "city", "one", "plus",
    "first", "new", "united", "power", "energy", "water", "gas", "utilities",
    "software", "cloud", "systems", "foods", "food", "kitchen", "eats",
    "university", "college", "institute", "school", "academy", "motors",
    "auto", "automotive", "cars", "department", "office", "council", "digital",
    "county", "municipal", "books", "polytechnic", "distribution", "eats", "water",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "at", "by", "for", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those", "you",
    "your", "yours", "i", "me", "my", "we", "our", "they", "their", "them",
    "he", "she", "his", "her", "not", "no", "can", "could", "will", "would",
    "should", "may", "might", "must", "do", "does", "did", "has", "have", "had",
    "so", "only", "also", "any", "all", "each", "per", "up", "out", "into",
    "within", "after", "before", "than", "more", "less", "over", "under",
    "there", "here", "what", "which", "who", "when", "where", "why", "how",
    "about", "via", "yes", "both", "such", "other",
}
FILLERS = {"uh", "um", "uhh", "umm", "er", "erm", "like", "so", "okay", "ok", "hi", "hey", "hello", "yeah", "well"}

NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")
CAP_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’\-]*[A-Za-z0-9]\b|\b[A-Z]\b")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
MARKDOWN_RE = re.compile(r"(^|\s)(#{1,6}\s|\*\*|__|\|.*\||```)")


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Issue:
    level: str  # "ERROR" | "WARN"
    where: str
    msg: str

    def __str__(self) -> str:
        return f"{self.level:5s} {self.where}: {self.msg}"


@dataclass
class OrgBundle:
    domain: str
    org_slug: str
    path: Path
    docs: list = field(default_factory=list)
    items: list = field(default_factory=list)

    @property
    def org_name(self) -> str:
        names = Counter(d.get("org") for d in self.docs + self.items if d.get("org"))
        return names.most_common(1)[0][0] if names else self.org_slug


class Report:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def error(self, where: str, msg: str) -> None:
        self.issues.append(Issue("ERROR", where, msg))

    def warn(self, where: str, msg: str) -> None:
        self.issues.append(Issue("WARN", where, msg))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "WARN"]


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def n_words(text: str) -> int:
    return len(text.split())


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def sha_int(key: str) -> int:
    return int(hashlib.sha256((SPLIT_SALT + key).encode("utf-8")).hexdigest(), 16)


def norm_number(tok: str) -> str:
    """Normalise a numeric token: drop thousands separators, keep decimals/times."""
    tok = tok.strip(".,:")
    # Thousands separators: 1,499 / 1,00,000 -> 1499 / 100000
    if re.fullmatch(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?", tok):
        tok = tok.replace(",", "")
    return tok


def numbers_in(text: str) -> set:
    out = set()
    for m in NUM_RE.finditer(text):
        tok = norm_number(m.group(0))
        if not tok:
            continue
        out.add(tok)
        # "6:30" also implies "6" and "30"; "1,499" stays whole.
        if ":" in tok or "." in tok:
            out.update(p for p in re.split(r"[:.]", tok) if p)
    return out


def number_tokens_strict(text: str) -> list:
    """Numbers that a key fact asserts (whole tokens, normalised)."""
    return [norm_number(m.group(0)) for m in NUM_RE.finditer(text) if norm_number(m.group(0))]


def content_words(text: str) -> list:
    words = [w.lower().replace("’", "'") for w in WORD_RE.findall(text)]
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()]


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def capitalised_terms(text: str) -> list:
    """Capitalised tokens that are not sentence-initial (names, plan names, months)."""
    out = []
    for sent in re.split(r"(?<=[.!?;:])\s+|\s+[-–—]\s+|\(|\)", text):
        sent = sent.strip()
        for i, m in enumerate(CAP_RE.finditer(sent)):
            tok = m.group(0)
            if m.start() == 0:
                continue  # sentence-initial capital carries no signal
            if tok.lower() in STOPWORDS or tok in {"I", "I'm", "I've", "I'd", "I'll"}:
                continue
            out.append(tok)
    return out


def normalise_question(q: str) -> str:
    words = [w.lower().replace("’", "'") for w in WORD_RE.findall(q)]
    return " ".join(w for w in words if w not in FILLERS)


def org_name_tokens(org: str) -> list:
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", org)]
    return [t for t in toks if t not in GENERIC_NAME_WORDS and len(t) >= 3]


def contains_word(text: str, word: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


HISTORY_ROLE_MAP = {"user": "user", "caller": "user", "assistant": "assistant", "agent": "assistant"}


def normalise_turn(t):
    """Return the canonical {"role": "user"|"assistant", "content": str} turn, or None if malformed.

    Raw files may use either {"role": "caller"|"agent", "text"} (part A authoring)
    or the canonical {"role": "user"|"assistant", "content"} shape (CONTRACTS §2).
    """
    if not isinstance(t, dict):
        return None
    role = HISTORY_ROLE_MAP.get(t.get("role"))
    content = t.get("content", t.get("text"))
    if role is None or not isinstance(content, str) or not content.strip():
        return None
    return {"role": role, "content": content}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path, report: Report) -> list:
    rows = []
    if not path.exists():
        report.error(str(path), "file missing")
        return rows
    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            if not line.strip():
                report.warn(f"{path}:{ln}", "blank line skipped")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                report.error(f"{path}:{ln}", f"invalid JSON: {exc}")
                continue
            if not isinstance(obj, dict):
                report.error(f"{path}:{ln}", "line is not a JSON object")
                continue
            obj["_line"] = ln
            rows.append(obj)
    return rows


def load_raw(raw_dir: Path, domains: list | None, report: Report) -> list:
    bundles = []
    if not raw_dir.exists():
        report.error(str(raw_dir), "raw directory does not exist")
        return bundles
    for ddir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if domains and ddir.name not in domains:
            continue
        for odir in sorted(p for p in ddir.iterdir() if p.is_dir()):
            b = OrgBundle(domain=ddir.name, org_slug=odir.name, path=odir)
            b.docs = read_jsonl(odir / "docs.jsonl", report)
            b.items = read_jsonl(odir / "items.jsonl", report)
            bundles.append(b)
    if domains:
        missing = set(domains) - {b.domain for b in bundles}
        for d in sorted(missing):
            report.error(f"raw/{d}", "requested domain has no org directories")
    return bundles


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_docs(b: OrgBundle, report: Report, all_doc_ids: dict) -> None:
    where0 = f"{b.domain}/{b.org_slug}/docs"
    n = len(b.docs)
    if not (DOC_COUNT_RANGE[0] <= n <= DOC_COUNT_RANGE[1]):
        report.error(where0, f"{n} docs (spec: {DOC_COUNT_RANGE[0]}-{DOC_COUNT_RANGE[1]})")
    prefix = f"d2-{b.domain}-{b.org_slug}-"
    orgs = {d.get("org") for d in b.docs}
    if len(orgs) > 1:
        report.error(where0, f"inconsistent org names in docs: {sorted(map(str, orgs))}")
    for d in b.docs:
        where = f"{where0}:{d['_line']}"
        keys = set(d) - {"_line"}
        if DOC_KEYS - keys:
            report.error(where, f"missing keys {sorted(DOC_KEYS - keys)}")
        if keys - DOC_KEYS:
            report.warn(where, f"extra keys {sorted(keys - DOC_KEYS)} (kept)")
        did = d.get("doc_id", "")
        if not isinstance(did, str) or not did.startswith(prefix):
            report.error(where, f"doc_id {did!r} must start with {prefix!r}")
        if did in all_doc_ids:
            report.error(where, f"duplicate doc_id {did!r} (also in {all_doc_ids[did]})")
        all_doc_ids[did] = where
        if d.get("dataset") != DATASET:
            report.error(where, f"dataset must be {DATASET!r}")
        if d.get("domain") != b.domain:
            report.error(where, f"domain {d.get('domain')!r} != directory {b.domain!r}")
        text = d.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            report.error(where, "empty text")
            continue
        nw = n_words(text)
        if not (DOC_WORD_RANGE[0] <= nw <= DOC_WORD_RANGE[1]):
            report.error(where, f"{did}: {nw} words (spec: {DOC_WORD_RANGE[0]}-{DOC_WORD_RANGE[1]})")
        if MARKDOWN_RE.search(text):
            report.warn(where, f"{did}: text looks like markdown")
        if not (d.get("title") or "").strip():
            report.error(where, f"{did}: empty title")
        if not d.get("source") or not d.get("licence"):
            report.error(where, f"{did}: source/licence missing")


def validate_items(b: OrgBundle, report: Report, all_org_names: list) -> None:
    where0 = f"{b.domain}/{b.org_slug}/items"
    docs_by_id = {d.get("doc_id"): d for d in b.docs}
    org = b.org_name
    name_toks = org_name_tokens(org)
    seen_ids = set()
    type_counts = Counter()
    n_hist = 0

    for it in b.items:
        where = f"{where0}:{it['_line']}({it.get('id')})"
        keys = set(it) - {"_line"}
        if ITEM_KEYS - keys:
            report.error(where, f"missing keys {sorted(ITEM_KEYS - keys)}")
        extra = keys - ITEM_KEYS
        if "split" in extra:
            report.warn(where, "'split' in raw item is ignored (builder assigns it)")
            extra.discard("split")
        if extra:
            report.warn(where, f"extra keys {sorted(extra)} (kept)")
        lid = it.get("id")
        if not isinstance(lid, str) or not lid:
            report.error(where, "missing local id")
        elif lid in seen_ids:
            report.error(where, f"duplicate local id {lid!r}")
        seen_ids.add(lid)
        if it.get("dataset") != DATASET:
            report.error(where, f"dataset must be {DATASET!r}")
        if it.get("domain") != b.domain:
            report.error(where, f"domain {it.get('domain')!r} != directory {b.domain!r}")
        if it.get("org") != org:
            report.error(where, f"org {it.get('org')!r} != {org!r}")

        itype = it.get("item_type")
        if itype not in ITEM_TYPES:
            report.error(where, f"item_type {itype!r} not in {ITEM_TYPES}")
            continue
        type_counts[itype] += 1
        needs = itype in ITEM_TYPES_RETRIEVAL
        if it.get("retrieval_needed_authored") is not needs:
            report.error(where, f"retrieval_needed_authored must be {needs} for {itype}")

        q = it.get("question")
        if not isinstance(q, str) or not q.strip():
            report.error(where, "empty question")
            continue
        qn = n_words(q)
        if not (QUESTION_WORD_RANGE[0] <= qn <= QUESTION_WORD_RANGE[1]):
            report.warn(where, f"question has {qn} words")
        if MARKDOWN_RE.search(q) or "*" in q:
            report.error(where, "markdown in question")
        ea = it.get("expected_answer")
        if not isinstance(ea, str) or not ea.strip():
            report.error(where, "empty expected_answer")
            ea = ""
        al = it.get("answer_aliases")
        if not isinstance(al, list) or not all(isinstance(a, str) and a.strip() for a in al):
            report.error(where, "answer_aliases must be a list of non-empty strings")
        if it.get("popularity") is not None:
            report.warn(where, "popularity should be null for d2_support")

        # History
        hist = it.get("history")
        hist_text = ""
        if not isinstance(hist, list):
            report.error(where, "history must be a list")
            hist = []
        norm_hist = []
        for t in hist:
            nt = normalise_turn(t)
            if nt is None:
                report.error(where, f"bad history turn {t!r} (need {{role: user|assistant, content}})")
            else:
                norm_hist.append(nt)
                hist_text += " " + nt["content"]
        if hist:
            n_hist += 1
            if norm_hist and norm_hist[-1]["role"] != "assistant":
                report.warn(where, "history should end with an assistant (agent) turn")

        # Organisation name must not be spoken by the caller (or the agent in history)
        spoken = q + " " + hist_text
        if contains_word(spoken, org):
            report.error(where, f"question/history names the organisation {org!r}")
        else:
            for tok in name_toks:
                if contains_word(spoken, tok):
                    report.error(where, f"question/history contains org-name token {tok!r}")
        for other in all_org_names:
            if other != org and contains_word(spoken, other):
                report.error(where, f"question/history names another organisation {other!r}")

        # Meta / key facts
        meta = it.get("meta")
        if not isinstance(meta, dict):
            report.error(where, "meta must be an object")
            meta = {}
        kf = meta.get("key_facts")
        if not isinstance(kf, list) or not kf or not all(isinstance(k, str) and k.strip() for k in kf):
            report.error(where, "meta.key_facts must be a non-empty list of strings")
            kf = []
        note = str(meta.get("note", "")).strip()
        if itype == "counter_prior" and not (str(meta.get("common_prior", "")).strip() or note):
            report.error(where, "counter_prior needs meta.common_prior (or meta.note)")
        if itype == "ambiguous" and not (str(meta.get("plausible_wrong_answer", "")).strip() or note):
            report.error(where, "ambiguous needs meta.plausible_wrong_answer (or meta.note)")

        gold = it.get("gold_doc_ids")
        if not isinstance(gold, list):
            report.error(where, "gold_doc_ids must be a list")
            gold = []
        sup = meta.get("superseded_doc_id")
        if sup is not None:
            if sup not in docs_by_id:
                report.error(where, f"superseded_doc_id {sup!r} not in this org's docs")
            if sup in gold:
                report.error(where, "superseded doc listed as gold")

        if not needs:
            if gold:
                report.error(where, "general item must have empty gold_doc_ids")
            check_general_answer(it, b, kf, ea, where, report)
            continue

        if not gold:
            report.error(where, f"{itype} item needs gold_doc_ids")
            continue
        missing = [g for g in gold if g not in docs_by_id]
        if missing:
            report.error(where, f"gold doc(s) not in this org's docs: {missing}")
            continue
        gold_text = " ".join(docs_by_id[g].get("text", "") + " " + docs_by_id[g].get("title", "") for g in gold)
        check_fact_support(kf, ea, gold_text, where, report)

    n = len(b.items)
    if n != TARGET_ITEMS_PER_ORG:
        report.warn(where0, f"{n} items (spec target {TARGET_ITEMS_PER_ORG})")
    if n:
        for t, frac in TARGET_TYPE_FRACTIONS.items():
            got = type_counts[t] / n
            if abs(got - frac) > TYPE_FRACTION_TOLERANCE:
                report.warn(where0, f"{t}: {type_counts[t]}/{n} = {got:.0%} (target ~{frac:.0%})")
        hf = n_hist / n
        if not (HISTORY_FRACTION_RANGE[0] - 0.02 <= hf <= HISTORY_FRACTION_RANGE[1] + 0.02):
            report.warn(where0, f"{n_hist}/{n} items with history ({hf:.0%}; spec 10-15%)")


def check_fact_support(key_facts: list, expected: str, gold_text: str, where: str, report: Report) -> None:
    gold_nums = numbers_in(gold_text)
    gold_lower = gold_text.lower().replace("’", "'")
    gold_stems = {stem(w) for w in content_words(gold_text)}
    for fact in key_facts:
        for num in number_tokens_strict(fact):
            if num not in gold_nums:
                report.error(where, f"key fact number {num!r} not found in gold docs: {fact!r}")
        for term in capitalised_terms(fact):
            t = term.lower().replace("’", "'")
            if not contains_word(gold_lower, t):
                report.error(where, f"key fact name/term {term!r} not found in gold docs: {fact!r}")
        cw = content_words(fact)
        if cw:
            cover = sum(stem(w) in gold_stems for w in cw) / len(cw)
            if cover < 0.5:
                report.warn(where, f"key fact weakly supported ({cover:.0%} of content words in gold): {fact!r}")
        elif not number_tokens_strict(fact):
            report.warn(where, f"key fact has nothing checkable: {fact!r}")
    for num in number_tokens_strict(expected):
        if num not in gold_nums:
            report.warn(where, f"expected_answer number {num!r} not in gold docs")


def check_general_answer(it: dict, b: OrgBundle, key_facts: list, expected: str, where: str, report: Report) -> None:
    """Heuristic: a general answer should not lean on values that are specific to this org's KB."""
    kb_text = " ".join(d.get("text", "") for d in b.docs)
    kb_nums = numbers_in(kb_text)
    common = {str(i) for i in range(0, 13)} | {"24", "30", "48", "60", "72", "90", "100", "365", "1000"}
    text = expected + " " + " ".join(key_facts)
    nums = sorted({n for n in number_tokens_strict(text) if n in kb_nums and n not in common})
    if nums:
        report.warn(where, f"general answer uses value(s) {nums} that also appear in the org KB; make sure they are not org-specific")
    kb_lower = kb_text.lower()
    allow = {t.lower() for t in GENERAL_TERM_ALLOW} | {m.lower() for m in MONTHS}
    terms = sorted({t for t in capitalised_terms(text) if t.lower() not in allow and contains_word(kb_lower, t.lower())})
    if terms:
        # Plan/product names from the KB in a general answer are suspicious.
        report.warn(where, f"general answer mentions KB term(s) {terms}; make sure the answer is not org-specific")


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
# Capitalised terms that are domain-generic (acronyms, standards, places) and fine in general answers.
GENERAL_TERM_ALLOW = [
    "SIM", "eSIM", "PUK", "PIN", "IMEI", "APN", "4G", "5G", "LTE", "VoLTE", "Wi-Fi", "WiFi", "SMS", "OTP",
    "UPI", "IFSC", "SWIFT", "IBAN", "CVV", "ATM", "KYC", "PAN", "Aadhaar", "NEFT", "RTGS", "IMPS", "EMI",
    "NCB", "TPA", "OPD", "IPD", "PNR", "GST", "VAT", "India", "Indian", "UK", "US", "UAE", "EU", "RBI",
    "IRDAI", "TRAI", "DGCA", "ID", "PDF", "SOS", "HD", "Mbps", "GB", "MB", "OK", "COD", "IATA", "TV",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "GP", "NHS", "ECG",
]


def validate_pairs(b: OrgBundle, report: Report) -> None:
    where0 = f"{b.domain}/{b.org_slug}/pairs"
    groups = defaultdict(list)
    for it in b.items:
        pid = it.get("pair_id")
        if it.get("item_type") not in ITEM_TYPES:
            continue
        if pid in (None, ""):
            if it.get("item_type") in ITEM_TYPES_RETRIEVAL:
                report.error(f"{where0}({it.get('id')})", "retrieval-needed item has no pair_id")
            else:
                report.warn(f"{where0}({it.get('id')})", "general item has no pair_id (unpaired)")
            continue
        groups[pid].append(it)
    for pid, members in sorted(groups.items(), key=lambda kv: natural_key(kv[0])):
        gens = [m for m in members if m["item_type"] == "general"]
        rets = [m for m in members if m["item_type"] in ITEM_TYPES_RETRIEVAL]
        where = f"{where0}({pid})"
        if not gens:
            report.error(where, "pair group has no general item")
            continue
        if not rets:
            report.error(where, "pair group has no retrieval-needed item")
            continue
        if len(gens) > 1:
            report.error(where, f"{len(gens)} general items in one pair group (must be exactly one)")
        if len(rets) > 2:
            report.warn(where, f"{len(rets)} retrieval-needed items in one pair group (spec: 1-2)")
        glens = [n_words(g.get("question", "")) for g in gens]
        for r in rets:
            rl = n_words(r.get("question", ""))
            diff = min(abs(rl - gl) for gl in glens)
            if diff > PAIR_LENGTH_TOLERANCE:
                report.error(where, f"length mismatch: {r.get('id')} has {rl} words vs general {glens} (tolerance ±{PAIR_LENGTH_TOLERANCE})")
        norms = [normalise_question(m.get("question", "")) for m in members]
        if len(set(norms)) < len(norms):
            report.error(where, "identical questions inside a pair group")


def validate_dedup(bundles: list, report: Report) -> None:
    rows = []
    for b in bundles:
        for it in b.items:
            if isinstance(it.get("question"), str):
                norm = normalise_question(it["question"])
                rows.append((f"{b.domain}/{b.org_slug}", it.get("id"), it.get("pair_id"), norm, set(norm.split())))
    for i in range(len(rows)):
        oi, idi, pi, ni, si = rows[i]
        for j in range(i + 1, len(rows)):
            oj, idj, pj, nj, sj = rows[j]
            if oi == oj and pi is not None and pi == pj:
                continue  # pair partners are meant to share a template
            if not si or not sj:
                continue
            jac = len(si & sj) / len(si | sj)
            if jac < 0.5:
                continue
            ratio = difflib.SequenceMatcher(None, ni, nj).ratio()
            if ratio >= DEDUP_ERROR_RATIO:
                report.error(f"dedup", f"near-identical questions {oi}:{idi} / {oj}:{idj} (ratio {ratio:.2f})")
            elif ratio >= DEDUP_WARN_RATIO:
                report.warn(f"dedup", f"similar questions {oi}:{idi} / {oj}:{idj} (ratio {ratio:.2f})")


def validate(bundles: list, report: Report) -> None:
    all_doc_ids: dict = {}
    names = Counter()
    for b in bundles:
        names[b.org_name] += 1
    for name, c in names.items():
        if c > 1:
            report.error("orgs", f"organisation name {name!r} used by {c} org directories")
    all_org_names = sorted(names)
    for b in bundles:
        if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", b.org_slug):
            report.error(str(b.path), "org_slug must be lowercase ascii words joined by underscores")
        validate_docs(b, report, all_doc_ids)
        validate_items(b, report, all_org_names)
        validate_pairs(b, report)
    validate_dedup(bundles, report)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def domain_index(domains_present: list) -> dict:
    extra = sorted(d for d in domains_present if d not in DOMAIN_ORDER)
    order = DOMAIN_ORDER + extra
    return {d: order.index(d) + 1 for d in domains_present}


def assign_ids(bundles: list) -> list:
    """Return enriched items with deterministic global ids.

    id      = d2_support-DDOSS  (DD domain index, O org index within domain, SS item sequence)
    pair_id = d2-pair-DDOPP     (PP pair sequence within org)

    Domain index comes from DOMAIN_ORDER; org index from the sorted org slugs of
    that domain; item / pair sequence from the natural sort of local ids.
    """
    didx = domain_index(sorted({b.domain for b in bundles}))
    by_domain = defaultdict(list)
    for b in bundles:
        by_domain[b.domain].append(b)
    out = []
    for dom, orgs in by_domain.items():
        orgs.sort(key=lambda b: b.org_slug)
        if len(orgs) > 9:
            raise SystemExit(f"domain {dom} has {len(orgs)} orgs; id scheme supports <= 9")
        for oi, b in enumerate(orgs, 1):
            items = sorted(b.items, key=lambda it: natural_key(it.get("id", "")))
            if len(items) > 99:
                raise SystemExit(f"{dom}/{b.org_slug} has {len(items)} items; id scheme supports <= 99")
            local_pairs = sorted({it["pair_id"] for it in items if it.get("pair_id")}, key=natural_key)
            pmap = {p: f"d2-pair-{didx[dom]:02d}{oi}{k:02d}" for k, p in enumerate(local_pairs, 1)}
            label = DOMAIN_LABELS.get(dom, dom.replace("_", " "))
            for si, it in enumerate(items, 1):
                meta = dict(it.get("meta") or {})
                meta.update({
                    "local_id": it.get("id"),
                    "local_pair_id": it.get("pair_id"),
                    "org_slug": b.org_slug,
                    "domain_label": label,
                    "system_prompt": SYSTEM_PROMPT_TEMPLATE.format(org=b.org_name, domain=label),
                    "question_words": n_words(it.get("question", "")),
                })
                rec = {
                    "id": f"{DATASET}-{didx[dom]:02d}{oi}{si:02d}",
                    "dataset": DATASET,
                    "domain": dom,
                    "org": it.get("org"),
                    "question": it.get("question"),
                    "expected_answer": it.get("expected_answer"),
                    "answer_aliases": it.get("answer_aliases", []),
                    "gold_doc_ids": it.get("gold_doc_ids", []),
                    "item_type": it.get("item_type"),
                    "retrieval_needed_authored": it.get("retrieval_needed_authored"),
                    "pair_id": pmap.get(it.get("pair_id")),
                    "popularity": None,
                    "history": [normalise_turn(t) for t in it.get("history", [])],
                    "split": None,
                    "meta": meta,
                }
                out.append(rec)
    out.sort(key=lambda r: r["id"])
    return out


def group_stratum(members: list) -> str:
    types = {m["item_type"] for m in members}
    for t in TYPE_PRECEDENCE:
        if t in types:
            return t
    return "general"


def assign_splits(items: list) -> dict:
    """CONTRACTS §7: 30% dev / 70% test, deterministic, stratified, pairs together.

    Unit of assignment = pair group (global pair_id; an unpaired item is its own
    unit), so pair partners always share a split. Allocation is done per
    organisation: each org gets n_dev = round(0.30 * n_units) dev units, spread
    over the type strata (rarest retrieval-needed type in the group: ambiguous >
    counter_prior > org_specific) by largest remainder. Within a stratum, units
    are ordered by unsalted sha256(unit key) and the lowest-hash units are dev.
    Because allocation never looks outside an org, adding domains or orgs never
    changes the split of existing items.
    """
    units = defaultdict(list)
    for it in items:
        units[it["pair_id"] or it["id"]].append(it)
    by_org = defaultdict(lambda: defaultdict(list))
    for key, members in units.items():
        org_key = f"{members[0]['domain']}/{members[0]['meta']['org_slug']}"
        by_org[org_key][group_stratum(members)].append(key)
    dev_units = set()
    for org_key, strata in sorted(by_org.items()):
        n_units = sum(len(v) for v in strata.values())
        n_dev = int(math.floor(DEV_FRACTION * n_units + 0.5))
        quota = {s: DEV_FRACTION * len(v) for s, v in strata.items()}
        alloc = {s: int(math.floor(q)) for s, q in quota.items()}
        rest = n_dev - sum(alloc.values())
        order = sorted(strata, key=lambda s: (-(quota[s] - alloc[s]), sha_int(f"{org_key}:{s}")))
        for s in order[:max(rest, 0)]:
            alloc[s] += 1
        for s, keys in strata.items():
            keys.sort(key=lambda k: (sha_int(k), k))
            dev_units.update(keys[:alloc[s]])
    for key, members in units.items():
        for it in members:
            it["split"] = "dev" if key in dev_units else "test"
    return {
        "dev": sorted(it["id"] for it in items if it["split"] == "dev"),
        "test": sorted(it["id"] for it in items if it["split"] == "test"),
        "seed": None,
        "method": (
            "Deterministic, stratified, pair-preserving split (CONTRACTS §7). Unit = pair group "
            "(global pair_id; unpaired items use their own id), so pair partners always share a split. "
            "Per organisation, n_dev = round(0.30 * n_units) units are allocated to dev across type strata "
            "(rarest retrieval-needed type in the group: ambiguous > counter_prior > org_specific) by "
            "largest remainder; within a stratum the units with the lowest unsalted sha256(unit key) are dev. "
            "Allocation is per organisation, so adding domains or organisations never changes existing splits."
        ),
        "dev_fraction_target": DEV_FRACTION,
    }


def build_corpus(bundles: list) -> list:
    didx = domain_index(sorted({b.domain for b in bundles}))
    docs = []
    for b in sorted(bundles, key=lambda b: (didx[b.domain], b.org_slug)):
        for d in sorted(b.docs, key=lambda d: natural_key(d.get("doc_id", ""))):
            rec = {k: d.get(k) for k in ["doc_id", "dataset", "domain", "org", "title", "text", "source", "licence"]}
            if str(rec["licence"]).replace("-", " ").strip().upper() == "CC BY 4.0":
                rec["licence"] = "CC BY 4.0"
            docs.append(rec)
    return docs


def compute_stats(items: list, corpus: list, report: Report) -> dict:
    def tally(key_fn):
        c = defaultdict(Counter)
        for it in items:
            c[key_fn(it)][it["item_type"]] += 1
            c[key_fn(it)]["total"] += 1
            c[key_fn(it)][it["split"]] += 1
        return {k: dict(v) for k, v in sorted(c.items())}

    qlen = defaultdict(list)
    for it in items:
        qlen["retrieval_needed" if it["retrieval_needed_authored"] else "general"].append(it["meta"]["question_words"])
    pair_sizes = Counter(len([i for i in items if i["pair_id"] == p]) for p in {i["pair_id"] for i in items if i["pair_id"]})
    doc_words = [n_words(d["text"]) for d in corpus]
    return {
        "n_items": len(items),
        "n_docs": len(corpus),
        "n_orgs": len({(i["domain"], i["meta"]["org_slug"]) for i in items}),
        "n_pair_groups": len({i["pair_id"] for i in items if i["pair_id"]}),
        "pair_group_sizes": dict(sorted(pair_sizes.items())),
        "by_type": dict(Counter(i["item_type"] for i in items)),
        "by_split": dict(Counter(i["split"] for i in items)),
        "by_type_split": {t: dict(Counter(i["split"] for i in items if i["item_type"] == t)) for t in ITEM_TYPES},
        "by_domain": tally(lambda i: i["domain"]),
        "by_org": tally(lambda i: f"{i['domain']}/{i['org']}"),
        "with_history": sum(1 for i in items if i["history"]),
        "question_words": {k: {"mean": round(statistics.mean(v), 2), "sd": round(statistics.pstdev(v), 2),
                               "min": min(v), "max": max(v)} for k, v in qlen.items() if v},
        "doc_words": {"mean": round(statistics.mean(doc_words), 1), "min": min(doc_words), "max": max(doc_words)} if doc_words else {},
        "docs_per_org": dict(Counter(f"{d['domain']}/{d['org']}" for d in corpus)),
        "validator": {"errors": len(report.errors), "warnings": len(report.warnings)},
    }


# --------------------------------------------------------------------------- #
# Human verification sheet
# --------------------------------------------------------------------------- #
def gold_excerpt(item: dict, docs_by_id: dict, max_chars: int = 600) -> str:
    if not item["gold_doc_ids"]:
        return ""
    facts = " ".join(item["meta"].get("key_facts", [])) + " " + item["expected_answer"]
    fnums = set(number_tokens_strict(facts))
    fwords = {stem(w) for w in content_words(facts)}
    scored = []
    for gid in item["gold_doc_ids"]:
        d = docs_by_id.get(gid)
        if not d:
            continue
        for k, s in enumerate(re.split(r"(?<=[.!?])\s+", d["text"])):
            score = 3 * len(numbers_in(s) & fnums) + len({stem(w) for w in content_words(s)} & fwords)
            scored.append((score, gid, k, s))
    best = sorted(scored, key=lambda x: (-x[0], x[1], x[2]))[:3]
    best.sort(key=lambda x: (x[1], x[2]))
    out, cur = [], None
    for _, gid, _, s in best:
        out.append((f"[{gid}] " if gid != cur else "") + s)
        cur = gid
    text = " ... ".join(out)
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def write_verification_sample(items: list, corpus: list, path: Path, frac: float = 0.20) -> int:
    docs_by_id = {d["doc_id"]: d for d in corpus}
    strata = defaultdict(list)
    for it in items:
        strata[(it["domain"], it["item_type"])].append(it)
    chosen = []
    for key, members in sorted(strata.items()):
        members.sort(key=lambda it: (sha_int("verify:" + it["id"]), it["id"]))
        k = max(1, int(math.floor(frac * len(members) + 0.5)))
        chosen.extend(members[:k])
    chosen.sort(key=lambda it: it["id"])
    cols = ["id", "split", "domain", "org", "item_type", "retrieval_needed_authored", "pair_id",
            "history", "question", "expected_answer", "key_facts", "prior_or_plausible_wrong",
            "gold_doc_ids", "gold_doc_excerpt", "label_correct (Y/N)", "answer_correct (Y/N)", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for it in chosen:
            m = it["meta"]
            hist = " | ".join(f"{t['role']}: {t['content']}" for t in it["history"])
            w.writerow([
                it["id"], it["split"], it["domain"], it["org"], it["item_type"],
                it["retrieval_needed_authored"], it["pair_id"] or "", hist, it["question"],
                it["expected_answer"], " ; ".join(m.get("key_facts", [])),
                m.get("common_prior") or m.get("plausible_wrong_answer") or m.get("note") or "",
                " ".join(it["gold_doc_ids"]), gold_excerpt(it, docs_by_id), "", "", "",
            ])
    return len(chosen)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_summary(stats: dict) -> None:
    print(f"\nitems={stats['n_items']} docs={stats['n_docs']} orgs={stats['n_orgs']} "
          f"pair_groups={stats['n_pair_groups']} sizes={stats['pair_group_sizes']} with_history={stats['with_history']}")
    print(f"by_type={stats['by_type']}  by_split={stats['by_split']}")
    hdr = f"{'domain/org':48s} {'total':>5s} {'org_sp':>6s} {'ctr_pr':>6s} {'ambig':>6s} {'gener':>6s} {'dev':>4s} {'test':>4s}"
    print(hdr)
    for k, v in list(stats["by_domain"].items()) + [("--", {})] + list(stats["by_org"].items()):
        if k == "--":
            print("-" * len(hdr))
            continue
        print(f"{k[:48]:48s} {v.get('total', 0):5d} {v.get('org_specific', 0):6d} {v.get('counter_prior', 0):6d} "
              f"{v.get('ambiguous', 0):6d} {v.get('general', 0):6d} {v.get('dev', 0):4d} {v.get('test', 0):4d}")
    print(f"question_words={stats['question_words']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--domains", nargs="*", default=None, help="restrict to these domain directories")
    ap.add_argument("--check-only", action="store_true", help="validate, print issues, write nothing")
    ap.add_argument("--force", action="store_true", help="write outputs even if the validator reports errors")
    ap.add_argument("--quiet-warnings", action="store_true", help="print only errors")
    args = ap.parse_args(argv)
    domains = [d for part in (args.domains or []) for d in part.split(",") if d] or None

    report = Report()
    bundles = load_raw(args.raw_dir, domains, report)
    validate(bundles, report)

    for iss in report.issues:
        if iss.level == "ERROR" or not args.quiet_warnings:
            print(iss)
    print(f"\nvalidator: {len(report.errors)} errors, {len(report.warnings)} warnings "
          f"({len(bundles)} orgs, {sum(len(b.items) for b in bundles)} items, {sum(len(b.docs) for b in bundles)} docs)")

    if args.check_only:
        return 1 if report.errors else 0
    if report.errors and not args.force:
        print("refusing to write outputs with validator errors (use --force to override)")
        return 1
    if not bundles:
        print("nothing to build")
        return 1

    items = assign_ids(bundles)
    splits = assign_splits(items)
    corpus = build_corpus(bundles)
    stats = compute_stats(items, corpus, report)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "items.jsonl", items)
    write_jsonl(out / "corpus.jsonl", corpus)
    (out / "splits.json").write_text(json.dumps(splits, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "validation_report.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"validator: {len(report.errors)} errors, {len(report.warnings)} warnings\n")
        for iss in report.issues:
            fh.write(str(iss).replace(str(args.raw_dir), "raw") + "\n")
    n_ver = write_verification_sample(items, corpus, out / "verification_sample.csv")
    print_summary(stats)
    print(f"\nwrote {out}/items.jsonl ({len(items)}), corpus.jsonl ({len(corpus)}), splits.json, stats.json, "
          f"validation_report.txt, verification_sample.csv ({n_ver})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
