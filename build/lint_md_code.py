#!/usr/bin/env python3
"""Lint Python code blocks in README.md files for indentation errors.

Scans aiinfra/daily/week*/day*/README.md, extracts ```python fenced code
blocks, dedents common leading whitespace, and runs ast.parse() to detect
IndentationError / TabError. Other SyntaxErrors are ignored because some
code blocks are intentionally incomplete pseudocode.

Usage:
    python3 build/lint_md_code.py

Exits 1 if any indentation errors are found, 0 otherwise.
"""

import ast
import re
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = REPO_ROOT / "aiinfra" / "daily"

OPEN_FENCE_RE = re.compile(r"^(\s*)```python\b")
CLOSE_FENCE_RE = re.compile(r"^\s*```")


def extract_python_blocks(text: str):
    """Yield (code_start_line, code) for each ```python fenced block.

    code_start_line is the 1-indexed line number of the first code line
    (the line immediately after the opening ```python fence).
    """
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if OPEN_FENCE_RE.match(lines[i]):
            fence_line = i + 1  # 1-indexed line of opening fence
            i += 1
            code_lines = []
            while i < n and not CLOSE_FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            code = "\n".join(code_lines)
            yield fence_line + 1, code
        else:
            i += 1


def lint_file(path: Path) -> tuple:
    """Return (num_blocks, errors) for python blocks in the given file.

    errors is a list of (relative_path_str, line_no, message).
    """
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(REPO_ROOT)
    num_blocks = 0
    errors = []
    for code_start_line, code in extract_python_blocks(text):
        num_blocks += 1
        dedented = textwrap.dedent(code)
        try:
            ast.parse(dedented)
        except IndentationError as e:
            # TabError is a subclass of IndentationError.
            err_line = code_start_line + (e.lineno - 1) if e.lineno else code_start_line
            errors.append((str(rel), err_line, f"{type(e).__name__}: {e.msg}"))
        except SyntaxError:
            pass  # other syntax errors — likely pseudocode, skip
    return num_blocks, errors


def main() -> int:
    files = sorted(DAILY_DIR.glob("week*/day*/README.md"))
    total_blocks = 0
    all_errors = []
    for path in files:
        num_blocks, errors = lint_file(path)
        total_blocks += num_blocks
        all_errors.extend(errors)

    for rel, line, msg in all_errors:
        print(f"{rel}:{line}: {msg}")

    print(f"Checked {len(files)} files, {total_blocks} code blocks, {len(all_errors)} errors")
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
