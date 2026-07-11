#!/usr/bin/env python3
"""Convert week1 SVGs from GitHub dark theme to hand-drawn sketch style."""
import re
import sys
from pathlib import Path

# Color mapping: dark theme → light sketch theme
COLOR_MAP = {
    # backgrounds
    "#0d1117": "#fdfcf7",
    "#161b22": "#fffdf5",
    "#1f2937": "#f0ede4",
    # text
    "#c9d1d9": "#2b2b2b",
    "#8b949e": "#666666",
    # borders
    "#30363d": "#3a3a3a",
    "#484f58": "#555555",
    # green
    "#238636": "#4a9d5f",
    "#3fb950": "#5cb86c",
    "#56d364": "#6cdb74",
    # blue
    "#1f6feb": "#4a8edb",
    "#58a6ff": "#6ab0ff",
    "#6c9fff": "#7cafff",
    # red
    "#f85149": "#e57373",
    "#ff7b72": "#ef8b84",
    # yellow/gold
    "#d29922": "#e8a838",
    "#e3b341": "#f0c050",
    # purple
    "#8957e5": "#9b6dd9",
    "#a371f7": "#b689e8",
}

SKETCH_FILTER_FULL = '''  <defs>
    <filter id="sketch" x="-3%" y="-3%" width="106%" height="106%">
      <feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves="2" seed="4" result="n"/>
      <feDisplacementMap in="SourceGraphic" in2="n" scale="1.8"/>
    </filter>
  </defs>
'''

SKETCH_FILTER_ONLY = '''    <filter id="sketch" x="-3%" y="-3%" width="106%" height="106%">
      <feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves="2" seed="4" result="n"/>
      <feDisplacementMap in="SourceGraphic" in2="n" scale="1.8"/>
    </filter>
'''

SKETCH_FONT = "'Comic Sans MS','Marker Felt','Segoe UI',cursive"


def convert_svg(path: Path):
    text = path.read_text(encoding="utf-8")

    # 1. Color replacements (case-insensitive)
    for old, new in COLOR_MAP.items():
        # fill attributes
        text = text.replace(f'fill="{old}"', f'fill="{new}"')
        text = text.replace(f'fill="{old.upper()}"', f'fill="{new}"')
        text = text.replace(f'fill="{old.lower()}"', f'fill="{new}"')
        # stroke attributes
        text = text.replace(f'stroke="{old}"', f'stroke="{new}"')
        text = text.replace(f'stroke="{old.upper()}"', f'stroke="{new}"')
        text = text.replace(f'stroke="{old.lower()}"', f'stroke="{new}"')
        # stop-color in gradients
        text = text.replace(f'stop-color:{old}', f'stop-color:{new}')
        text = text.replace(f'stop-color:{old.upper()}', f'stop-color:{new}')

    # 2. Add font-family to root svg if not present
    if "font-family" not in text[:200]:
        text = re.sub(
            r'(<svg[^>]*?)>',
            r'\1 font-family="' + SKETCH_FONT + '">',
            text,
            count=1,
        )
    else:
        # replace existing font-family
        text = re.sub(
            r"font-family='[^']*'",
            f"font-family='{SKETCH_FONT}'",
            text,
        )
        text = re.sub(
            r'font-family="[^"]*"',
            f'font-family="{SKETCH_FONT}"',
            text,
        )

    # 3. Inject sketch filter def + wrapping group
    # Find position after the opening <svg ...> tag
    svg_open_end = text.find(">", text.find("<svg")) + 1
    if svg_open_end == 0:
        print(f"  WARN: could not find <svg> in {path}")
        return False

    # Check if there's already a <defs> — if so, inject filter element into it
    defs_start = text.find("<defs>", svg_open_end)
    if defs_start != -1 and defs_start < svg_open_end + 500:
        # Insert filter element inside existing defs (not a new defs wrapper)
        defs_end = text.find(">", defs_start) + 1
        text = text[:defs_end] + "\n" + SKETCH_FILTER_ONLY + text[defs_end:]
    else:
        # Insert new defs after svg open
        text = text[:svg_open_end] + "\n" + SKETCH_FILTER_FULL + text[svg_open_end:]

    # 4. Wrap content in <g filter="url(#sketch)">
    # Find the closing </svg>
    svg_close = text.rfind("</svg>")
    if svg_close == -1:
        print(f"  WARN: could not find </svg> in {path}")
        return False

    # Insert wrapping group right after defs
    # Find end of defs block
    defs_close = text.find("</defs>", svg_open_end)
    if defs_close != -1:
        insert_pos = defs_close + len("</defs>") + 1  # +1 for newline
    else:
        insert_pos = svg_open_end

    text = (
        text[:insert_pos]
        + '\n  <g filter="url(#sketch)">\n'
        + text[insert_pos:svg_close]
        + "  </g>\n"
        + text[svg_close:]
    )

    path.write_text(text, encoding="utf-8")
    return True


def main():
    img_dir = Path("week1/website/images")
    svgs = sorted(img_dir.glob("*.svg"))
    print(f"Found {len(svgs)} SVGs to convert")
    success = 0
    for svg in svgs:
        ok = convert_svg(svg)
        if ok:
            success += 1
            print(f"  OK: {svg.name}")
        else:
            print(f"  FAIL: {svg.name}")
    print(f"\nConverted {success}/{len(svgs)} SVGs to sketch style")


if __name__ == "__main__":
    main()
