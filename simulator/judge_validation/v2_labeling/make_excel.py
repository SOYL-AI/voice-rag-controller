"""
make_excel.py — build one Excel workbook per annotator from the r3 labelling
sheets (sheet_annotator_A.csv / sheet_annotator_B.csv), and read completed
workbooks back into the CSV format merge_sheets.py expects.

    python -m simulator.judge_validation.v2_labeling.make_excel build --out-dir <folder>
    python -m simulator.judge_validation.v2_labeling.make_excel collect \
        --xlsx <folder>/Annotator_A_labeling.xlsx --out sheet_annotator_A_done.csv

Each workbook has three tabs:
  * Instructions: the r3 rule from LABELING_GUIDE.md, the worked examples and
    the borderline rules.
  * Labeling: the same 300 rows, in the same order as the CSV. Only the yellow
    Verdict and Notes columns are editable. Verdict is a correct/incorrect
    dropdown.
  * Progress: live counts computed by formulas.

The judge verdicts are never written into the workbooks: they come only from
the CSV sheets, which already hide them.
"""

import argparse
import csv
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
FONT = "Arial"
YELLOW = PatternFill("solid", fgColor="FFF2CC")
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
GREY = PatternFill("solid", fgColor="F2F2F2")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")

# (CSV column, header shown to the annotator, width, editable)
COLUMNS = [
    ("row", "Row", 6, False),
    ("company", "Company", 16, False),
    ("domain", "Industry", 12, False),
    ("conversation_before", "Earlier conversation", 34, False),
    ("question", "Caller's question", 34, False),
    ("reference_answer", "Reference answer", 40, False),
    ("core_facts", "CORE facts (must all be stated)", 34, False),
    ("supplementary_facts", "Supplementary facts (may be left out, must not be contradicted)", 30, False),
    ("candidate_answer", "Agent's candidate answer (judge this)", 48, False),
    ("verdict", "Verdict", 13, True),
    ("notes", "Notes (optional)", 24, True),
    ("case_id", "Case ID (do not edit)", 30, False),
]

RULE_TEXT = [
    ("How to label (rubric r3)", True),
    ("Each row is a caller's question to a customer-support voice agent at a fictional company. "
     "Read the earlier conversation (if any), the reference answer, its CORE facts and SUPPLEMENTARY facts, "
     "then judge the agent's candidate answer.", False),
    ("", False),
    ("Fill in ONLY the yellow columns on the Labeling tab: Verdict (pick from the dropdown) and Notes (optional).", True),
    ("", False),
    ("The candidate is CORRECT only if BOTH hold:", True),
    ("1. It states every CORE fact. Any wording, equivalent units/formats and close approximations count: "
     "\"₹50\" = \"50 rupees\" = \"Rs 50\" = \"about ₹50\"; \"48 hours\" = \"two days\"; \"9 AM\" = \"09:00\"; \"13 amp\" = \"13 A\".", False),
    ("2. It contradicts nothing in the reference or the supplementary facts (no different figure, time, name, price, "
     "place, condition, limit or exception).", False),
    ("Supplementary facts may be left out; they only count against the answer when it contradicts them.", False),
    ("", False),
    ("Otherwise it is INCORRECT. In particular:", True),
    ("• Deferring without stating the core facts (\"please contact us\", \"check your policy\", \"it depends on your plan\").", False),
    ("• Declining, \"I don't know\", or answering a different question.", False),
    ("• A vague or hedged answer that leaves out a core fact, even if nothing it says is false.", False),
    ("Extra true detail is fine. Wording, length, tone, politeness, apologies and naming the company do not matter.", False),
    ("Judge only what the answer SAYS. If you think the reference itself is wrong, still judge against it and say so in Notes.", False),
    ("", False),
    ("Borderline cases", True),
    ("• Ranges: \"about ₹50\" for ₹50 is correct; \"₹40–60\" is incorrect (it does not state the figure).", False),
    ("• Hedges with the facts present: \"I believe it's ₹50\" is correct.", False),
    ("• Broader statement (\"bring ID\" where the core fact says \"original ID\"): correct unless it changes the meaning; note it.", False),
    ("• \"Why\" questions with several reasons: correct ONLY if the answer gives the CORE reason. "
     "If you disagree with which reason was made core, still follow the rule and write \"core reason?\" in Notes.", False),
    ("• Unsure: pick your best verdict and write \"unsure\" in Notes. Never leave Verdict blank.", False),
    ("", False),
    ("Ground rules: work on your own (no comparing with the other annotator), do not look up company documents, "
     "do not use any AI tool. About 2 hours. Save the file when done and send it back unchanged in name.", True),
]

EXAMPLES = [
    ("A replacement SIM costs ₹50", "Issued in-store with original ID", "\"It's about fifty rupees for a new SIM.\"", "correct",
     "Core fact stated as an approximation; supplementary left out."),
    ("A replacement SIM costs ₹50", "Issued in-store with original ID", "\"It's ₹50, and you can get it delivered to your home.\"", "incorrect",
     "Home delivery contradicts \"issued in-store\"."),
    ("A replacement SIM costs ₹50", "Issued in-store with original ID", "\"Please check our website for the current SIM replacement charges.\"",
     "incorrect", "Defers without the core fact."),
    ("A replacement SIM costs ₹50", "Issued in-store with original ID", "\"A replacement SIM costs ₹100.\"", "incorrect", "Contradicts the core fact."),
    ("Burst pipes are covered · an excess of AED 500 applies", "Not covered if the home was empty more than 60 days",
     "\"Our policy typically covers burst pipes, so you should be fine.\"", "incorrect", "Hedge: a core fact (the excess) is missing."),
    ("Burst pipes are covered · an excess of AED 500 applies", "Not covered if the home was empty more than 60 days",
     "\"Yes, burst pipes are covered, with a 500-dirham excess.\"", "correct", "Both core facts stated."),
    ("A washing machine needs a 13 amp fuse", "A fuse that blows again suggests a fault", "\"It should be a 13 A fuse.\"", "correct",
     "Core fact stated; supplementary advice left out."),
    ("A washing machine needs a 13 amp fuse", "A fuse that blows again suggests a fault", "\"Use a 5 amp fuse for that.\"", "incorrect",
     "Contradicts the core fact."),
]


def _font(**kw):
    return Font(name=FONT, **kw)


def _instructions(wb: Workbook, who: str) -> None:
    ws = wb.active
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 150
    ws["A1"] = f"Answer-correctness labelling — Annotator {who}"
    ws["A1"].font = _font(bold=True, size=14)
    r = 3
    for text, bold in RULE_TEXT:
        cell = ws.cell(row=r, column=1, value=text)
        cell.font = _font(bold=bold, size=11)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="Worked examples (fictional company \"Kestrel Mobile\") — these are NOT rows to label").font = _font(bold=True, size=11)
    r += 1
    heads = ["Core facts", "Supplementary facts", "Candidate answer", "Verdict", "Why"]
    widths = [40, 34, 52, 11, 46]
    for c, (h, w) in enumerate(zip(heads, widths), start=2):
        cell = ws.cell(row=r, column=c, value=h)
        cell.font = _font(bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.border = BOX
        ws.column_dimensions[cell.column_letter].width = w
    for ex in EXAMPLES:
        r += 1
        for c, v in enumerate(ex, start=2):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = _font(size=10)
            cell.alignment = WRAP_TOP
            cell.border = BOX
    ws.sheet_view.showGridLines = False


def _labeling(wb: Workbook, rows: list) -> str:
    ws = wb.create_sheet("Labeling")
    for c, (_, head, width, editable) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=c, value=head)
        cell.font = _font(bold=True, color="FFFFFF" if not editable else "000000")
        cell.fill = YELLOW if editable else HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BOX
        ws.column_dimensions[cell.column_letter].width = width
    ws.row_dimensions[1].height = 48

    for i, row in enumerate(rows, start=2):
        for c, (key, _, _, editable) in enumerate(COLUMNS, start=1):
            value = row.get(key, "")
            if key == "row":
                value = int(value)
            if key in ("verdict", "notes"):
                value = None  # always start blank
            cell = ws.cell(row=i, column=c, value=value)
            cell.font = _font(size=10)
            cell.alignment = WRAP_TOP
            cell.border = BOX
            cell.fill = YELLOW if editable else (GREY if i % 2 else PatternFill())
            cell.protection = Protection(locked=not editable)

    last = len(rows) + 1
    verdict_col = next(cell.column_letter for cell in ws[1] if cell.value == "Verdict")
    dv = DataValidation(type="list", formula1='"correct,incorrect"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Invalid verdict",
                        error="Choose correct or incorrect from the dropdown.")
    ws.add_data_validation(dv)
    dv.add(f"{verdict_col}2:{verdict_col}{last}")

    rng = f"{verdict_col}2:{verdict_col}{last}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"correct"'],
                                                  fill=PatternFill("solid", fgColor="C6EFCE")))
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"incorrect"'],
                                                  fill=PatternFill("solid", fgColor="FFC7CE")))
    ws.freeze_panes = "E2"
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(COLUMNS)).column_letter}{last}"
    ws.protection.sheet = True
    ws.protection.formatColumns = False
    ws.protection.formatRows = False
    ws.protection.autoFilter = False
    ws.protection.sort = True
    return verdict_col


def _progress(wb: Workbook, verdict_col: str, n: int) -> None:
    ws = wb.create_sheet("Progress")
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 14
    rng = f"Labeling!${verdict_col}$2:${verdict_col}${n + 1}"
    rows = [
        ("Rows to label", n, None),
        ("Labelled", f'=COUNTIF({rng},"correct")+COUNTIF({rng},"incorrect")', None),
        ("  of which correct", f'=COUNTIF({rng},"correct")', None),
        ("  of which incorrect", f'=COUNTIF({rng},"incorrect")', None),
        ("Remaining", "=B2-B3", None),
        ("Done", "=IF(B2=0,0,B3/B2)", "0.0%"),
    ]
    ws["A1"] = "Progress (updates automatically)"
    ws["A1"].font = _font(bold=True, size=12)
    for r, (label, value, fmt) in enumerate(rows, start=2):
        ws.cell(row=r, column=1, value=label).font = _font(size=11)
        cell = ws.cell(row=r, column=2, value=value)
        cell.font = _font(size=11, bold=(label == "Remaining"))
        if fmt:
            cell.number_format = fmt
    ws.cell(row=2, column=3, value="Source: the 300 sampled answers in this kit (kit_manifest.json).").font = _font(size=9, italic=True)
    ws.conditional_formatting.add("B6", FormulaRule(formula=["$B$6>0"], fill=PatternFill("solid", fgColor="FFC7CE")))


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for who in ("A", "B"):
        with open(HERE / f"sheet_annotator_{who}.csv", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        wb = Workbook()
        _instructions(wb, who)
        verdict_col = _labeling(wb, rows)
        _progress(wb, verdict_col, len(rows))
        wb.active = 1  # open on the Labeling tab
        wb.calculation.fullCalcOnLoad = True  # Progress formulas compute when opened
        path = out_dir / f"Annotator_{who}_labeling.xlsx"
        wb.save(path)
        print(f"wrote {path} ({len(rows)} rows)")


def collect(xlsx: Path, out: Path) -> None:
    """Completed workbook -> CSV with the original sheet columns, for merge_sheets.py."""
    wb = load_workbook(xlsx, data_only=True)
    ws = wb["Labeling"]
    header = [c.value for c in ws[1]]
    index = {head: header.index(head) for _, head, _, _ in COLUMNS}
    fields = [key for key, _, _, _ in COLUMNS]
    blanks = 0
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[index["Case ID (do not edit)"]] in (None, ""):
                continue
            rec = {key: ("" if row[index[head]] is None else row[index[head]]) for key, head, _, _ in COLUMNS}
            rec["verdict"] = str(rec["verdict"]).strip().lower()
            blanks += rec["verdict"] == ""
            writer.writerow(rec)
    print(f"wrote {out}; blank verdicts: {blanks}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--xlsx", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.out_dir)
    else:
        collect(args.xlsx, args.out)


if __name__ == "__main__":
    main()
