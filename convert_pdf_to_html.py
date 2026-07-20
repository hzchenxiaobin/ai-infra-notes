#!/usr/bin/env python3
"""Convert a PDF file to a single HTML file using PyMuPDF."""

import sys
from pathlib import Path

import fitz


def convert_pdf_to_html(pdf_path: str, html_path: str) -> None:
    doc = fitz.open(pdf_path)
    pages_html = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pages_html.append(
            f'<div class="page" id="page-{page_num + 1}">\n'
            f'{page.get_text("html")}\n'
            f'</div>\n'
        )
    doc.close()

    full_html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '    <meta charset="UTF-8">\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"    <title>{Path(pdf_path).stem}</title>\n"
        "    <style>\n"
        "        body { font-family: Georgia, serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; color: #222; }\n"
        "        .page { border-bottom: 2px solid #ccc; margin-bottom: 40px; padding-bottom: 20px; }\n"
        "        p { margin: 0.6em 0; }\n"
        "        h1, h2, h3, h4, h5, h6 { margin: 1em 0 0.5em; }\n"
        "    </style>\n"
        "</head>\n"
        "<body>\n"
        + "\n".join(pages_html)
        + "</body>\n"
        "</html>\n"
    )

    Path(html_path).write_text(full_html, encoding="utf-8")
    print(f"Converted {pdf_path} -> {html_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: convert_pdf_to_html.py <input.pdf> <output.html>")
        sys.exit(1)
    convert_pdf_to_html(sys.argv[1], sys.argv[2])
