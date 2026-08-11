#!/usr/bin/env python3
"""Course consistency checks for aiinfra/daily content.

Three checks, run over aiinfra/daily/week*/**/*.md (plan/archive is excluded):

1. duplicate-title  : headings whose text contains a repeated half
                      (copy-paste bug, e.g. "## Day 1：ABC...ABC"),
                      including pure-ASCII inline repeats and same-section
                      headings that appear more than once in one file.
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
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = REPO_ROOT / "aiinfra" / "daily"

# We only scan aiinfra/daily/week*/**/*.md.  aiinfra/daily/plan/archive/ holds
# historical 8-week plans and is intentionally not scanned (it would
# legitimately contain old terms and stale internal paths).
EXCLUDED_RELPATHS = [re.compile(r"^plan/archive/")]


def is_excluded(rel: Path) -> bool:
    s = str(rel).replace("\\", "/")
    return any(rx.search(s) for rx in EXCLUDED_RELPATHS)

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
WORD_RE = re.compile(r"[A-Za-z0-9_]")
WORD_TAIL_RE = re.compile(r"[A-Za-z0-9_]$")

# Heading levels up to 6 because some deep sub-sections (e.g. #####) exist.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Inline repeat heuristics:
#   - adjacent copies like "XX" or "X / X" (covers pure-ASCII paste bugs)
#   - repeated suffix containing >=4 CJK chars (covers "调度对比...调度对比")
ADJ_RE = re.compile(r"(.{6,}?)\1")
ADJ_SEP_RE = re.compile(r"(.{6,}?)[\s·/—–:：、|-]{1,3}\1")
INLINE_DUP_MIN_CJK = 6
INLINE_DUP_MIN_ASCII = 10
CJK_SUFFIX_MIN_CHARS = 4
SECTION_DUP_MIN_LEN = 10


def _word_extended(text: str, chunk: str, pos: int) -> bool:
    """True if the occurrence of chunk at pos is followed by a word char,
    i.e. the chunk is merely a prefix of a longer identifier."""
    end = pos + len(chunk)
    return end < len(text) and bool(WORD_RE.match(text[end]))


def _inline_duplicate_kind(text: str) -> str | None:
    """Classify a single heading line that looks copy-pasted."""
    # 1) Any >=10-char substring containing CJK that appears more than once.
    #    Guard against identifier-prefix artifacts such as
    #    "为什么用 SequenceGroup 而不是直接用 Sequence？" where the repeated
    #    substring "用 Sequence" is just a prefix of "SequenceGroup".
    for size in range(len(text) // 2, 9, -1):
        for start in range(0, len(text) - 2 * size + 1):
            chunk = text[start : start + size]
            if not CJK_RE.search(chunk):
                continue
            j = text.find(chunk, start + 1)
            if j == -1:
                continue
            if WORD_TAIL_RE.search(chunk) and (
                _word_extended(text, chunk, start) or _word_extended(text, chunk, j)
            ):
                continue
            return "inline"

    # 2) Adjacent repeated copies: "NVIDIA CUDA vs Ascend CANNNVIDIA CUDA vs
    #    Ascend CANN" or "Title —— SubTitleSubTitle".
    for rx in (ADJ_RE, ADJ_SEP_RE):
        m = rx.search(text)
        if m:
            chunk = m.group(1)
            min_len = INLINE_DUP_MIN_CJK if CJK_RE.search(chunk) else INLINE_DUP_MIN_ASCII
            if len(chunk) >= min_len:
                return "inline"

    # 3) Repeated suffix with >=4 CJK chars: e.g. the trailing "调度对比"
    #    appears twice in one heading.  We deliberately do NOT apply this to
    #    pure-ASCII suffixes because headings like "A vs no_A" are legitimate.
    for size in range(len(text) // 2, 0, -1):
        suffix = text[-size:]
        if not suffix.strip():
            continue
        if text[:-size].find(suffix) != -1 and len(CJK_RE.findall(suffix)) >= CJK_SUFFIX_MIN_CHARS:
            return "inline"

    return None


def find_duplicate_titles(path: Path):
    """Flag two classes of duplicate-heading bugs:

    a) A single heading line that repeats itself (inline copy-paste),
       including pure-ASCII adjacent repeats like
       "NVIDIA CUDA vs Ascend CANNNVIDIA CUDA vs Ascend CANN".
    b) The same section heading appearing more than once at the same level
       in one file.  Short generic names ("观察重点", "今日总结") are
       intentionally ignored by a length threshold to avoid false positives.
    """
    findings = []
    headings_by_level = defaultdict(list)

    for i, line in iter_body_lines(path):
        m = HEADING_RE.match(line)
        if not m:
            continue
        level, raw = m.groups()
        text = raw.strip()

        kind = _inline_duplicate_kind(text)
        if kind:
            findings.append((i, text, kind))

        norm = re.sub(r"\s+", " ", text).strip()
        headings_by_level[(level, norm)].append(i)

    for (level, norm), lines in headings_by_level.items():
        if len(lines) >= 2 and len(norm) >= SECTION_DUP_MIN_LEN:
            for line_no in lines[1:]:
                findings.append((line_no, norm, "section"))

    findings.sort(key=lambda x: x[0])
    return findings


# ---------------------------------------------------------------- check 2
# Curated stale patterns from the old 8-week course. NOTE: "10 周算法面试
# 刷题计划" / "10-week-plan" is a legitimate external LeetCode plan name and
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
    # Old 8-week framing used in W10D3 and other migration leftovers.
    (re.compile(r"Week 1-7"), "Week 1-7"),
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
    for i, line in iter_body_lines(path):
        for m in LINK_RE.finditer(line):
            target = m.group(1).strip().split()[0] if m.group(1).strip() else ""
            if not target or target.startswith(("http://", "https://", "#", "mailto:", "(")):
                continue
            if target.startswith("...") and not target.startswith(".../"):
                continue
            target = target.split("#")[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                findings.append((i, target))
    return findings


# ---------------------------------------------------------------- driver
def main():
    baseline_only = "--baseline" in sys.argv
    md_files = [
        p for p in sorted(DAILY_DIR.glob("week*/**/*.md"))
        if not is_excluded(p.relative_to(DAILY_DIR))
    ]
    total = 0
    for path in md_files:
        rel = path.relative_to(REPO_ROOT)
        for line_no, text, kind in find_duplicate_titles(path):
            tag = "dup-title-inline" if kind == "inline" else "dup-title-section"
            print(f"[{tag}] {rel}:{line_no}: {text[:70]}")
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
