#!/usr/bin/env python3
"""
Build the paper reading website from aiinfra/paper/.

Generates:
  - aiinfra/paper/website/index.html: paper list overview
  - aiinfra/paper/website/<paper>/index.html: per-paper note page (renders README.md)
  - aiinfra/paper/website/images/: copied SVG assets
  - PDFs are copied next to each paper's index.html for direct download.

This builder is copied into public/paper/ by the root build.py, so asset links
use root_prefix="../" for the overview and "../../" for per-paper pages.
"""

import re
import shutil
from pathlib import Path
from typing import Optional


PAPER_DIR = Path(__file__).parent.parent
WEBSITE_DIR = Path(__file__).parent
IMAGES_DIR = PAPER_DIR / "images"


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    text = text.replace("</script>", "\\x3c/script>")
    return text


def rewrite_readme_paths(markdown_text: str) -> str:
    """Rewrite local paths in README.md so they work after deployment.

    README.md lives in aiinfra/paper/<paper>/ and references:
      - ../SKILL.md  -> paper reading home (../index.html)
      - ../images/   -> stays as ../images/ (images are copied to public/paper/images/)
      - *.pdf        -> stays as-is (PDF copied next to index.html)
    """
    markdown_text = re.sub(
        r"\]\(\.\./SKILL\.md\)",
        "](../index.html)",
        markdown_text,
    )
    return markdown_text


def extract_title(markdown_text: str, fallback: str) -> str:
    """Extract title from the first H1 heading, fallback to directory name."""
    match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def find_papers() -> list:
    """Discover paper subdirectories under aiinfra/paper/."""
    papers = []
    skip = {"images", "website", "__pycache__"}
    for subdir in sorted(PAPER_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name in skip:
            continue
        readme = subdir / "README.md"
        pdfs = sorted(subdir.glob("*.pdf"))
        if not readme.exists() and not pdfs:
            continue
        title = subdir.name.replace("_", " ")
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            title = extract_title(text, title)
            markdown = rewrite_readme_paths(text)
        else:
            markdown = None
        papers.append({
            "slug": subdir.name,
            "title": title,
            # 侧边栏只显示主标题（"——" 前半段），避免长副标题被省略号截断
            "short_title": title.split("——")[0].strip(),
            "dir": subdir,
            "readme": readme if readme.exists() else None,
            "pdfs": pdfs,
            "markdown": markdown,
        })
    return papers


def build_nav(papers: list, current_slug: Optional[str], root_prefix: str) -> str:
    """Build sidebar navigation for the paper site."""
    home_active = ' active' if current_slug is None else ''
    lines = [
        f'<a class="nav-link{home_active}" href="{root_prefix}paper/index.html">📌 论文列表</a>',
        '<div class="nav-section-title">论文笔记</div>',
    ]
    for paper in papers:
        active = ' active' if current_slug == paper["slug"] else ''
        if paper["readme"]:
            href = f"{root_prefix}paper/{paper['slug']}/index.html"
        elif paper["pdfs"]:
            # No README: link directly to the first PDF
            href = f"{root_prefix}paper/{paper['slug']}/{paper['pdfs'][0].name}"
        else:
            continue
        lines.append(f'<a class="nav-link{active}" href="{href}">{paper["short_title"]}</a>')
    return "\n".join(lines)


def page_template(title: str, page_title: str, root_prefix: str, nav: str,
                  markdown: Optional[str] = None, extra_content: Optional[str] = None) -> str:
    """Generate a paper site HTML page."""
    if markdown is not None:
        markdown_block = f'''<script type="text/markdown" id="markdown-source">
{escape_for_template_string(markdown)}
</script>'''
    else:
        markdown_block = ""

    if extra_content:
        content_html = extra_content
    else:
        content_html = ""

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="{root_prefix}css/style.css?v=4">
    <script src="{root_prefix}js/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="{root_prefix}js/markdown-math.js"></script>
    <link href="{root_prefix}css/prism-tomorrow.min.css" rel="stylesheet">
    <script src="{root_prefix}js/prism.min.js"></script>
    <script src="{root_prefix}js/prism-c.min.js"></script>
    <script src="{root_prefix}js/prism-cpp.min.js"></script>
    <script>Prism.languages.cuda=Prism.languages.extend("c",{{builtin:/\\b(?:__global__|__device__|__host__|__shared__|__constant__|__managed__|__restrict__|__syncthreads|__threadfence|__threadfence_block|blockIdx|threadIdx|blockDim|gridDim|warpSize)\\b/}});</script>
    <script src="{root_prefix}js/prism-bash.min.js"></script>
    <script src="{root_prefix}js/prism-python.min.js"></script>
</head>
<body>
    <button class="menu-toggle" aria-label="Toggle menu">☰</button>

    <div class="site-container">
        <aside class="sidebar">
            <div class="sidebar-header">
                <a href="{root_prefix}paper/index.html" style="text-decoration: none;">
                    <h1 class="sidebar-title">论文精读</h1>
                </a>
            </div>
            <nav class="sidebar-nav">
{nav}
            </nav>
        </aside>
        <main class="main-content">
            <div class="content-header">
                <h1>{page_title}</h1>
            </div>
            <div class="content" id="content">
{content_html}
            </div>
            {markdown_block}
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    const renderer = new marked.Renderer();
                    renderer.heading = function(text, level, raw) {{
                        let anchor = raw.toLowerCase()
                            .replace(/[^\\w\\s-]/g, '')
                            .replace(/\\s+/g, '-')
                            .replace(/-+/g, '-')
                            .replace(/^-|-$/g, '');
                        if (anchor && level >= 2) {{
                            return '<h' + level + ' id="' + anchor + '">' + text + '</h' + level + '>';
                        }}
                        return '<h' + level + '>' + text + '</h' + level + '>';
                    }};
                    marked.setOptions({{ renderer: renderer, headerIds: false, gfm: true, breaks: false, sanitize: false }});
                    const markdownSource = document.getElementById('markdown-source');
                    if (markdownSource) {{
                        try {{
                            document.getElementById('content').innerHTML = marked.parse(markdownSource.textContent);
                            if (window.Prism) Prism.highlightAll();
                        }} catch (err) {{
                            document.getElementById('content').innerHTML = '<div style="padding:20px;color:#ff7b72;background:#2d1515;border-radius:8px;"><h2>⚠️ 页面渲染失败</h2><p>' + err.message + '</p></div>';
                            console.error('Markdown render error:', err);
                        }}
                    }}
                }});
            </script>
        </main>
    </div>
    <script src="{root_prefix}js/main.js?v=5"></script>
</body>
</html>
'''


def build_index_content(papers: list, root_prefix: str) -> str:
    """Generate the markdown/HTML content for the paper list page."""
    lines = [
        "欢迎来到论文精读页面。这里收录了 AI Infra 相关的重要论文笔记与原文 PDF。",
        "",
        "## 论文列表",
        "",
        "| 论文 | 阅读笔记 | 原文 PDF |",
        "| --- | --- | --- |",
    ]
    for paper in papers:
        note_cell = "—"
        if paper["readme"]:
            note_cell = f'[阅读笔记]({root_prefix}paper/{paper["slug"]}/index.html)'
        pdf_cell = "<br>".join(
            f'[{pdf.name}]({root_prefix}paper/{paper["slug"]}/{pdf.name})'
            for pdf in paper["pdfs"]
        ) or "—"
        lines.append(f"| **{paper['title']}** | {note_cell} | {pdf_cell} |")
    return "\n".join(lines)


def clean_website_dir() -> None:
    """Remove old generated content but keep this build script and README."""
    if not WEBSITE_DIR.exists():
        WEBSITE_DIR.mkdir(parents=True)
        return
    for item in WEBSITE_DIR.iterdir():
        if item.name in {"build.py", "README.md"}:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main() -> None:
    clean_website_dir()

    papers = find_papers()
    print(f"Found {len(papers)} papers: {[p['slug'] for p in papers]}")

    # Copy shared images
    website_images = WEBSITE_DIR / "images"
    if IMAGES_DIR.exists():
        website_images.mkdir(parents=True)
        for img in IMAGES_DIR.iterdir():
            if img.is_file():
                shutil.copy2(img, website_images / img.name)
        print(f"Copied {len(list(IMAGES_DIR.iterdir()))} images to website/images/")

    # Generate overview page (deployed at public/paper/index.html)
    overview_nav = build_nav(papers, current_slug=None, root_prefix="../")
    overview_md = build_index_content(papers, root_prefix="../")
    overview_html = page_template(
        title="论文精读",
        page_title="论文精读",
        root_prefix="../",
        nav=overview_nav,
        markdown=overview_md,
    )
    (WEBSITE_DIR / "index.html").write_text(overview_html, encoding="utf-8")
    print("Generated paper/website/index.html")

    # Generate per-paper pages (deployed at public/paper/<slug>/index.html)
    for paper in papers:
        paper_web_dir = WEBSITE_DIR / paper["slug"]
        paper_web_dir.mkdir(parents=True)

        # Copy PDFs
        for pdf in paper["pdfs"]:
            shutil.copy2(pdf, paper_web_dir / pdf.name)

        if paper["readme"]:
            paper_nav = build_nav(papers, current_slug=paper["slug"], root_prefix="../../")
            paper_html = page_template(
                title=f"{paper['title']} - 论文精读",
                page_title=paper["title"],
                root_prefix="../../",
                nav=paper_nav,
                markdown=paper["markdown"],
            )
            (paper_web_dir / "index.html").write_text(paper_html, encoding="utf-8")
            print(f"Generated paper/website/{paper['slug']}/index.html")
        else:
            print(f"Skipped note page for {paper['slug']} (no README.md)")


if __name__ == "__main__":
    main()
