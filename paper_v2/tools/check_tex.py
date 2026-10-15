"""
Static checks for paper_v2/ when no TeX installation is available.

    PYTHONUTF8=1 python paper_v2/tools/check_tex.py

Checks, after stripping comments: brace balance per file, \\begin/\\end
balance and order, every \\ref/\\eqref has a \\label, every \\cite key is in
references.bib, every \\input and \\includegraphics target exists, the
bibliography has at most 15 entries, no stray carriage returns, and counts
the remaining \\todo / \\TBD / \\PEND markers. It is not a substitute for a
real compile; compile once in Overleaf.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def strip_comments(text):
    out = []
    for line in text.splitlines():
        # a % not preceded by a backslash starts a comment
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def collect(path, seen, errors):
    path = path if path.suffix else path.with_suffix(".tex")
    if path in seen:
        return ""
    seen.add(path)
    if not path.exists():
        errors.append(f"missing \\input target: {path.relative_to(ROOT)}")
        return ""
    raw = path.read_text(encoding="utf-8")
    if "\r" in raw.replace("\r\n", "\n"):
        errors.append(f"stray carriage return in {path.relative_to(ROOT)}")
    body = strip_comments(raw)
    depth = 0
    for i, ch in enumerate(body):
        if ch in "{}" and i > 0 and body[i - 1] == "\\":
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"unbalanced '}}' in {path.relative_to(ROOT)}")
                depth = 0
    if depth != 0:
        errors.append(f"{depth} unclosed '{{' in {path.relative_to(ROOT)}")
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
        kind, env = m.groups()
        if kind == "begin":
            stack.append(env)
        elif not stack or stack[-1] != env:
            errors.append(f"\\end{{{env}}} does not match {stack[-1:] or 'nothing'} in {path.relative_to(ROOT)}")
        else:
            stack.pop()
    if stack:
        errors.append(f"unclosed environments {stack} in {path.relative_to(ROOT)}")
    text = body
    for m in re.finditer(r"\\input\{([^}]+)\}", body):
        text += "\n" + collect(ROOT / m.group(1), seen, errors)
    return text


def main():
    errors = []
    text = collect(ROOT / "main.tex", set(), errors)
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    for ref in set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", text)):
        if ref not in labels:
            errors.append(f"undefined reference: {ref}")
    bib = (ROOT / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M)
    if len(keys) > 15:
        errors.append(f"{len(keys)} bibliography entries (limit 15)")
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        cited.update(k.strip() for k in group.split(","))
    for k in sorted(cited - set(keys)):
        errors.append(f"cited key not in references.bib: {k}")
    uncited = sorted(set(keys) - cited)
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        if not (ROOT / "figures" / g).exists():
            errors.append(f"missing figure file: figures/{g}")
    figs = len(re.findall(r"\\begin\{figure\*?\}", text))
    tabs = len(re.findall(r"\\begin\{table\*?\}", text))
    print(f"labels {len(labels)}, bib entries {len(keys)}, cited {len(cited)}, figures {figs}, tables {tabs}")
    if uncited:
        print("bib entries not yet cited in the outline:", ", ".join(uncited))
    for name in ("todo", "TBD", "PEND"):
        print(f"\\{name} markers outside comments: {len(re.findall(re.escape(chr(92) + name) + r'\{', text))}")
    for fig_label in sorted(l for l in labels if l.startswith(("fig:", "tab:"))):
        if not re.search(r"\\ref\{" + re.escape(fig_label) + r"\}", text):
            errors.append(f"{fig_label} is never referenced in the text")
    if errors:
        print("\nPROBLEMS:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("OK: no structural problems found")


if __name__ == "__main__":
    main()
