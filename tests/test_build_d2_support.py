"""Tests for datasets/v2/build_d2_support.py on a small synthetic raw tree."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_d2_support", ROOT / "datasets" / "v2" / "build_d2_support.py")
b2 = importlib.util.module_from_spec(_spec)
sys.modules["build_d2_support"] = b2
_spec.loader.exec_module(b2)

FILLER = ("Customers can reach the help desk by phone or chat and an agent will confirm "
          "the details on the account before making any change. ")


def _doc(domain, slug, org, n, extra):
    text = extra + " " + FILLER * 9
    return {"doc_id": f"d2-{domain}-{slug}-{n:02d}-topic", "dataset": "d2_support", "domain": domain,
            "org": org, "title": f"Topic {n}", "text": text, "source": "authored", "licence": "CC BY 4.0"}


def _item(domain, org, lid, pid, itype, q, gold=(), facts=("A fact",), hist=()):
    meta = {"key_facts": list(facts), "core_facts": list(facts)[:1]}
    if itype == "counter_prior":
        meta["common_prior"] = "the usual answer"
    if itype == "ambiguous":
        meta["plausible_wrong_answer"] = "a plausible answer"
    return {"id": lid, "dataset": "d2_support", "domain": domain, "org": org, "question": q,
            "expected_answer": "An answer.", "answer_aliases": ["x"], "gold_doc_ids": list(gold),
            "item_type": itype, "retrieval_needed_authored": itype != "general", "pair_id": pid,
            "popularity": None, "history": list(hist), "meta": meta}


# (general, retrieval-needed) question templates; distinct enough to pass the dedup check.
TEMPLATES = [
    ("Why would my card get declined when I'm travelling abroad?", "How many days ahead must I enable my card for abroad?"),
    ("What does a waiting period on a health plan actually mean?", "What's the waiting period on my plan for knee surgery?"),
    ("Can I keep my number if I move to a different network?", "Can I keep my number if I pause service for two months?"),
    ("Why's my broadband so much slower in the evenings?", "Do you compensate me when the broadband is down for a day?"),
    ("Is it safe to share the one-time password with an agent?", "Is there a charge for sending the statement by post?"),
    ("How much data does an hour of HD video use, roughly?", "How much data rolls over to next month on my plan?"),
    ("What's a PNR and where do I find it on the ticket?", "What's the fee to change the name on a ticket?"),
    ("Why does my ear pop when the plane is coming down?", "When does the check-in counter close for domestic flights?"),
    ("What's the difference between OPD and IPD at a hospital?", "What are the OPD timings at the city branch on Sundays?"),
    ("Why do I need to fast before a cholesterol test?", "How long before the test do I need to stop eating?"),
    ("What does cash on delivery mean for my order?", "How much extra do you add for cash on delivery?"),
    ("Why does the tracking page not update for a day sometimes?", "By what time do I need to order for next-day delivery?"),
]


def make_org(raw, domain, slug, org, n_pairs=6, offset=0):
    d = raw / domain / slug
    d.mkdir(parents=True)
    docs = [_doc(domain, slug, org, n, f"Refunds take {20 + n} working days at the Pune office.") for n in range(1, 14)]
    items = []
    k = 0
    for p in range(1, n_pairs + 1):
        k += 1
        items.append(_item(domain, org, f"q{k:02d}", f"p{p:02d}", "general",
                           TEMPLATES[(p + offset) % 12][0]))
        k += 1
        itype = ["org_specific", "counter_prior", "ambiguous"][p % 3]
        items.append(_item(domain, org, f"q{k:02d}", f"p{p:02d}", itype,
                           TEMPLATES[(p + offset) % 12][1],
                           gold=[docs[p - 1]["doc_id"]], facts=[f"{20 + p} working days at the Pune office"]))
    for path, rows in (("docs.jsonl", docs), ("items.jsonl", items)):
        with (d / path).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return docs, items


def test_clean_fixture_builds(tmp_path):
    raw = tmp_path / "raw"
    make_org(raw, "telecom", "alpha_tel", "Alpha Telco")
    make_org(raw, "telecom", "beta_tel", "Beta Telco", offset=6)
    out = tmp_path / "out"
    assert b2.main(["--raw-dir", str(raw), "--out-dir", str(out)]) == 0
    items = [json.loads(line) for line in (out / "items.jsonl").open(encoding="utf-8")]
    splits = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    assert len(items) == 24
    assert items[0]["id"] == "d2_support-01101"
    assert set(splits["dev"]) | set(splits["test"]) == {it["id"] for it in items}
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], set()).add(it["split"])
    assert all(len(s) == 1 for s in by_pair.values())
    assert "Alpha Telco" in items[0]["meta"]["system_prompt"]


def test_ids_and_splits_stable_when_domain_added(tmp_path):
    raw = tmp_path / "raw"
    make_org(raw, "telecom", "alpha_tel", "Alpha Telco")
    assert b2.main(["--raw-dir", str(raw), "--out-dir", str(tmp_path / "o1")]) == 0
    make_org(raw, "utilities", "gamma_power", "Gamma Grid", n_pairs=5, offset=7)
    assert b2.main(["--raw-dir", str(raw), "--out-dir", str(tmp_path / "o2")]) == 0
    a = {json.loads(l)["id"]: json.loads(l)["split"] for l in (tmp_path / "o1" / "items.jsonl").open(encoding="utf-8")}
    b = {json.loads(l)["id"]: json.loads(l)["split"] for l in (tmp_path / "o2" / "items.jsonl").open(encoding="utf-8")}
    assert all(b[i] == s for i, s in a.items())


def test_validator_catches_errors(tmp_path):
    raw = tmp_path / "raw"
    docs, items = make_org(raw, "telecom", "alpha_tel", "Alpha Telco")
    items[1]["question"] = "Alpha Telco, how long does a refund take to show up?"   # names org
    items[3]["meta"]["key_facts"] = ["99 working days"]                            # unsupported number
    items[5]["question"] = "How long " + "really " * 10 + "does it take?"           # length mismatch
    items[7]["meta"]["core_facts"] = ["a paraphrase that is not a key fact"]        # core not verbatim
    del items[9]["meta"]["core_facts"]                                               # core missing
    with (raw / "telecom" / "alpha_tel" / "items.jsonl").open("w", encoding="utf-8") as fh:
        for r in items:
            fh.write(json.dumps(r) + "\n")
    rep = b2.Report()
    b2.validate(b2.load_raw(raw, None, rep), rep)
    msgs = " | ".join(str(e) for e in rep.errors)
    assert "names the organisation" in msgs
    assert "'99' not found" in msgs
    assert "length mismatch" in msgs
    assert "core_facts not verbatim in key_facts" in msgs
    assert "core_facts must be a non-empty list" in msgs


def test_history_shapes_normalised_and_one_general_per_group(tmp_path):
    raw = tmp_path / "raw"
    docs, items = make_org(raw, "telecom", "alpha_tel", "Alpha Telco")
    items[0]["history"] = [{"role": "caller", "text": "I'm on the prepaid plan."},
                           {"role": "agent", "text": "Sure, what can I help with?"}]
    items[2]["history"] = [{"role": "user", "content": "I'm travelling next week."},
                           {"role": "assistant", "content": "Okay, go ahead."}]
    path = raw / "telecom" / "alpha_tel" / "items.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")
    out = tmp_path / "out"
    assert b2.main(["--raw-dir", str(raw), "--out-dir", str(out)]) == 0
    built = [json.loads(l) for l in (out / "items.jsonl").open(encoding="utf-8")]
    turns = [t for it in built for t in it["history"]]
    assert turns and all(set(t) == {"role", "content"} and t["role"] in ("user", "assistant") for t in turns)
    # a second general item in one pair group is an error
    items[2]["pair_id"] = items[0]["pair_id"]  # move p02's general item into p01
    path.write_text("".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")
    rep = b2.Report()
    b2.validate(b2.load_raw(raw, None, rep), rep)
    assert any("must be exactly one" in str(e) for e in rep.errors)
