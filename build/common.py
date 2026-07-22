"""Shared utilities for the website build system."""

import re
import shutil
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

PLAN_SOURCE = REPO_ROOT / "aiinfra" / "daily" / "plan" / "AI_Infra_8_week_plan_detailed.md"
COURSE_OVERVIEW_SOURCE = REPO_ROOT / "aiinfra" / "daily" / "README.md"
DAILY_DIR = REPO_ROOT / "aiinfra" / "daily"
STATIC_DIR = REPO_ROOT / "static"

DAY_TITLE_PATTERN = re.compile(r"^## Day (\d+)[：:]\s*(.+)$")


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    text = text.replace("</script>", "\\x3c/script>")
    return text


def extract_plan_weeks(plan_path: Path = None) -> list:
    """Extract week numbers and titles from the 8-week plan markdown."""
    if plan_path is None:
        plan_path = PLAN_SOURCE
    if not plan_path.exists():
        return []
    text = plan_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^##\s*[^\s]*\s*Week\s*(\d+)[:：]\s*(.+)$", re.MULTILINE)
    weeks = []
    for match in pattern.finditer(text):
        weeks.append({"num": int(match.group(1)), "title": match.group(2).strip()})
    return weeks


def get_day_info(week_dir: Path) -> list:
    """Return sorted day info [{'num': int, 'title': str}, ...] by parsing README titles."""
    info = []
    for day_dir in sorted(week_dir.glob("day*")):
        readme = day_dir / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        match = DAY_TITLE_PATTERN.match(first_line)
        if match:
            info.append({"num": int(match.group(1)), "title": match.group(2).strip()})
    return sorted(info, key=lambda d: d["num"])


def load_overview_and_days(week_dir: Path):
    """Load overview from weekN/README.md and per-day markdown from weekN/dayN/README.md.

    Returns (overview_text, days) where days is a list of
    {"num": int, "title": str, "markdown": str} sorted by day number.
    """
    readme_path = week_dir / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Week README not found: {readme_path}")
    overview = readme_path.read_text(encoding="utf-8")
    overview = re.sub(r"\]\((?:\.\./)?(?:website/)?images/", "](images/", overview)

    days = []
    for day_dir in sorted(week_dir.glob("day*")):
        readme = day_dir / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        text = re.sub(r"\]\((?:\.\./)?(?:website/)?images/", "](images/", text)
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        match = DAY_TITLE_PATTERN.match(first_line)
        if not match:
            raise ValueError(f"Cannot parse Day title from first line of {readme}: {first_line!r}")
        days.append({
            "num": int(match.group(1)),
            "title": match.group(2).strip(),
            "markdown": "\n".join(text.strip().splitlines()[1:]),
        })

    if not days:
        raise ValueError(f"No day*/README.md files found in {week_dir}")
    days.sort(key=lambda d: d["num"])
    return overview, days


def compute_root_prefix(output_path: Path, output_dir: Path) -> str:
    """Return the relative prefix (e.g. '../') from output_path's directory to output_dir."""
    try:
        rel_parent = output_path.parent.relative_to(output_dir)
    except ValueError:
        return ""
    depth = len(rel_parent.parts)
    return "../" * depth if depth > 0 else ""


def rewrite_md_links_to_html_weeks(markdown_text: str, root_prefix: str = "") -> str:
    """Rewrite local .md links to .html for GitHub Pages deployment (week pages).

    Links that escape the week directory (starting with ../../) are rewritten
    to relative paths from the generated page's location.
    """

    def replace_link(match):
        url = match.group(1)
        if not url.endswith(".md"):
            return match.group(0)
        new_url = url[:-3] + ".html"
        if new_url.endswith("README.html"):
            new_url = new_url[: -len("README.html")] + "index.html"
        if new_url.startswith("../../../../"):
            inner = new_url[len("../../../../"):]
            inner = re.sub(
                r"^leetgpu/week\d+/day\d+/(leetgpu-.*-solution\.html)$",
                r"leetgpu/\1",
                inner,
            )
            inner = re.sub(
                r"^leetcode/daily/week\d+/day\d+/([^/]+\.html)$",
                r"leetcode/problems/\1",
                inner,
            )
            new_url = root_prefix + inner
        return f"]({new_url})"

    return re.sub(r"\]\((?!https?://|#)([^)]+)\)", replace_link, markdown_text)


def build_day_cards_html(days: list, root_prefix: str = "") -> str:
    """Build the Day cards HTML block used on week overview pages."""
    html = '<div class="day-cards">\n'
    for day in days:
        html += (
            f'<a class="day-card" href="{root_prefix}day{day["num"]}.html">\n'
            f'  <div class="day-card-number">Day {day["num"]}</div>\n'
            f'  <div class="day-card-title">{day["title"]}</div>\n'
            f'</a>\n'
        )
    html += '</div>\n'
    return html


# ---------------------------------------------------------------------------
# Heading renderer JS snippets (inserted into the page <script> block)
# ---------------------------------------------------------------------------

HEADING_RENDERER_WEEKS = """renderer.heading = function(text, level, raw) {
            let anchor = raw.toLowerCase()
                .replace(/[^\\w\\s-]/g, '')
                .replace(/\\s+/g, '-')
                .replace(/-+/g, '-')
                .replace(/^-|-$/g, '');

            const dayMatch = raw.match(/^Day (\\d+)[:：]\\s*(.+)$/);
            if (dayMatch) {
                anchor = 'day-' + dayMatch[1];
            }

            if (level === 2 && anchor) {
                return '<h' + level + ' id="' + anchor + '">' + text + '</h' + level + '>';
            }
            return '<h' + level + '>' + text + '</h' + level + '>';
        };"""

HEADING_RENDERER_TOPICS = """renderer.heading = function(text, level, raw) {
            let anchor = raw.toLowerCase()
                .replace(/[^\\w\\s-]/g, '')
                .replace(/\\s+/g, '-')
                .replace(/-+/g, '-')
                .replace(/^-|-$/g, '');

            const dayMatch = raw.match(/^Day (\\d+)(?:[（(][^)）]*[）)])*[:：]\\s*(.+)$/);
            if (dayMatch) {
                anchor = 'day-' + dayMatch[1];
            }

            if (level === 2 && anchor) {
                return '<h' + level + ' id="' + anchor + '">' + text + '</h' + level + '>';
            }
            return '<h' + level + '>' + text + '</h' + level + '>';
        };"""


# ---------------------------------------------------------------------------
# Unified page template (standard layout: weeks, leetcode, leetgpu, topics)
# ---------------------------------------------------------------------------

def page_template(
    title: str,
    nav_html: str,
    markdown: str,
    *,
    root_prefix: str = "",
    page_title: Optional[str] = None,
    is_overview: bool = False,
    extra_scripts: str = "",
    sidebar_title: str = "AI Infra 8 周计划",
    sidebar_title_style: str = "",
    sidebar_href: Optional[str] = None,
    back_link_href: Optional[str] = None,
    show_back_link: bool = True,
    heading_renderer_js: str = "",
) -> str:
    """Generate a standard HTML page with sidebar navigation and markdown content."""
    escaped_markdown = escape_for_template_string(markdown)
    if page_title is None:
        page_title = title
    if sidebar_href is None:
        sidebar_href = f"{root_prefix}index.html"
    if back_link_href is None:
        back_link_href = f"{root_prefix}index.html"

    back_link = ""
    bottom_nav = ""
    if show_back_link and not is_overview:
        back_link = f'<a class="back-link" href="{back_link_href}">← 返回概览</a>'
        bottom_nav = f'<div class="day-nav-bottom"><a class="back-link" href="{back_link_href}">← 返回概览</a></div>'

    title_style_attr = f' style="{sidebar_title_style}"' if sidebar_title_style else ""

    if heading_renderer_js:
        renderer_block = f"        {heading_renderer_js}\n\n"
    else:
        renderer_block = ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <link rel="stylesheet" href="{root_prefix}css/style.css?v=4">
    <!-- Marked.js for Markdown rendering -->
    <script src="{root_prefix}js/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="{root_prefix}js/markdown-math.js"></script>
    <!-- Prism.js for syntax highlighting -->
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
                <a href="{sidebar_href}" style="text-decoration: none;">
                    <h1 class="sidebar-title"{title_style_attr}>{sidebar_title}</h1>
                </a>
            </div>
            <nav class="sidebar-nav">
{nav_html}
            </nav>
        </aside>

        <main class="main-content">
            <div class="page-header">
                <h1 class="page-title">{title}</h1>
                {back_link}
            </div>
            <article class="content" id="content"></article>
            {bottom_nav}
        </main>
    </div>

    <button class="back-to-top" aria-label="Back to top">↑</button>

    <script>
        const markdown = `{escaped_markdown}`;

        const renderer = new marked.Renderer();
{renderer_block}        marked.setOptions({{
            renderer: renderer,
            headerIds: false,
            gfm: true,
            breaks: false,
            sanitize: false
        }});

        try {{
            if (typeof marked === 'undefined') {{
                throw new Error('marked.js failed to load. Please check js/marked.min.js exists.');
            }}
            document.getElementById('content').innerHTML = marked.parse(markdown);

            if (window.Prism) {{
                Prism.highlightAll();
            }}
        }} catch (err) {{
            document.getElementById('content').innerHTML = '<div style="padding: 20px; color: #ff7b72; background: #2d1515; border-radius: 8px;">' +
                '<h2>⚠️ 页面渲染失败</h2>' +
                '<p>' + err.message + '</p>' +
                '<p>请打开浏览器控制台（Cmd + Option + J）查看详细错误。</p>' +
                '</div>';
            console.error('Markdown render error:', err);
        }}
    </script>
    {extra_scripts}
    <script src="{root_prefix}js/main.js?v=5"></script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Paper page template (uses <script type="text/markdown"> + DOMContentLoaded)
# ---------------------------------------------------------------------------

def paper_page_template(
    title: str,
    page_title: str,
    root_prefix: str,
    nav: str,
    markdown: Optional[str] = None,
    extra_content: Optional[str] = None,
) -> str:
    """Generate a paper site HTML page."""
    if markdown is not None:
        markdown_block = f'''<script type="text/markdown" id="markdown-source">
{escape_for_template_string(markdown)}
</script>'''
    else:
        markdown_block = ""

    content_html = extra_content or ""

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


def copy_static_assets(public_dir: Path) -> None:
    """Copy shared css/js from static/ to public/css/ and public/js/."""
    css_src = STATIC_DIR / "css"
    js_src = STATIC_DIR / "js"
    if css_src.exists():
        dst = public_dir / "css"
        dst.mkdir(parents=True, exist_ok=True)
        for item in css_src.iterdir():
            if item.is_file():
                shutil.copy2(item, dst / item.name)
    if js_src.exists():
        dst = public_dir / "js"
        dst.mkdir(parents=True, exist_ok=True)
        for item in js_src.iterdir():
            if item.is_file():
                shutil.copy2(item, dst / item.name)
