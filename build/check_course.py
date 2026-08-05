#!/usr/bin/env python3
"""Course consistency checks for aiinfra/daily content.

Three checks, run over aiinfra/daily/week*/**/*.md:

1. duplicate-title  : headings whose text contains a repeated half
                      (copy-paste bug, e.g. "## Day 1：ABC...ABC")
2. stale-terms      : leftover references from the old 8-week course
                      (curated pattern list, not a blanket grep)
3. dangling-links   : relative markdown links / image refs pointing to
                      files that do not exist

Usage:
    python3 build/check_course.py            # all checks, exit 1 on findings
    python3 build/check_course.py --baseline # print counts only, always exit 0
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = REPO_ROOT / "aiinfra" / "daily"

# ---------------------------------------------------------------- utils
FENCE_RE = re.compile(r"^\s*```")


def iter_body_lines(path: Path):
    """Yield (line_no, line) with fenced code blocks stripped."""
    in_fence = False
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield i, line


CJK_RE = re.compile(r"[一-鿿]")

# ---------------------------------------------------------------- check 1
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


def find_duplicate_titles(path: Path):
    """Flag headings containing a repeated >=10-char substring that includes
    CJK (the classic paste bug 'Title —— SubTitleSubTitle'). Pure-ASCII
    repeats like 'Sequence / SequenceGroup' are ignored on purpose."""
    findings = []
    for i, line in iter_body_lines(path):
        m = HEADING_RE.match(line)
        if not m:
            continue
        text = m.group(2)
        flagged = False
        for size in range(len(text) // 2, 9, -1):
            for start in range(0, len(text) - 2 * size + 1):
                chunk = text[start : start + size]
                if not CJK_RE.search(chunk):
                    continue
                if text.find(chunk, start + 1) != -1:
                    flagged = True
                    break
            if flagged:
                break
        if flagged:
            findings.append((i, text))
    return findings


# ---------------------------------------------------------------- check 2
# Curated stale patterns from the old 8-week course. NOTE: "8 周算法面试
# 刷题计划" / "8-week-plan" is a legitimate external LeetCode plan name and
# must NOT be flagged.
STALE_PATTERNS = [
    (re.compile(r"8 周能力地图"), "8 周能力地图"),
    (re.compile(r"8 周学习(?!计划)"), "8 周学习"),
    (re.compile(r"8 周完成"), "8 周完成"),
    (re.compile(r"8 周收官"), "8 周收官"),
    (re.compile(r"8 周总结"), "8 周总结"),
    (re.compile(r"Week 8 的第一天"), "Week 8 的第一天"),
    (re.compile(r"Week 8 完成标准"), "Week 8 完成标准"),
    (re.compile(r"Week 8 知识地图"), "Week 8 知识地图"),
    (re.compile(r"week8_summary"), "week8_summary"),
    (re.compile(r"aiinfra/week8"), "aiinfra/week8"),
]


def find_stale_terms(path: Path):
    findings = []
    for i, line in iter_body_lines(path):
        for rx, name in STALE_PATTERNS:
            if rx.search(line):
                findings.append((i, name, line.strip()[:80]))
    return findings


# ---------------------------------------------------------------- check 3
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
BARE_KERNEL_RE = re.compile(r"\]\((kernels/[^)]+)\)")


def find_dangling_links(path: Path):
    findings = []
    text = path.read_text(encoding="utf-8")
    for m in LINK_RE.finditer(text):
        target = m.group(1).strip().split()[0] if m.group(1).strip() else ""
        if not target or target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = target.split("#")[0]
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            line_no = text[: m.start()].count("\n") + 1
            findings.append((line_no, target))
    return findings


# ---------------------------------------------------------------- driver
def main():
    baseline_only = "--baseline" in sys.argv
    md_files = sorted(DAILY_DIR.glob("week*/**/*.md"))
    total = 0
    for path in md_files:
        rel = path.relative_to(REPO_ROOT)
        for line_no, text in find_duplicate_titles(path):
            print(f"[dup-title] {rel}:{line_no}: {text[:70]}")
            total += 1
        for line_no, name, snippet in find_stale_terms(path):
            print(f"[stale]     {rel}:{line_no}: ({name}) {snippet}")
            total += 1
        for line_no, target in find_dangling_links(path):
            print(f"[dangling]  {rel}:{line_no}: {target}")
            total += 1
    print(f"\nChecked {len(md_files)} files, {total} findings.")
    if baseline_only:
        return 0
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
