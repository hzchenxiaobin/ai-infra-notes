"""Shared utilities for the website build system."""

import html
import re
import shutil
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

GITHUB_REPO_URL = "https://github.com/hzchenxiaobin/ai-infra-notes"

PLAN_SOURCE = REPO_ROOT / "aiinfra" / "daily" / "plan" / "learning_plan_10week.md"
COURSE_OVERVIEW_SOURCE = REPO_ROOT / "aiinfra" / "daily" / "README.md"
DAILY_DIR = REPO_ROOT / "aiinfra" / "daily"
STATIC_DIR = REPO_ROOT / "static"

DAY_TITLE_PATTERN = re.compile(r"^## Day (\d+[a-z]?)[：:]\s*(.+)$")


def escape_for_template_string(text: str) -> str:
    """Escape a markdown string for embedding in a JS template string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    text = text.replace("</script>", "\\x3c/script>")
    return text


def extract_plan_weeks(plan_path: Path = None) -> list:
    """Extract week numbers and titles from the 10-week plan markdown."""
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
    """Return sorted day info [{'num': str, 'title': str}, ...] by parsing README titles.

    num is a string like "1" or "3b" (letter-suffixed supplementary days).
    Sorting is lexicographic, which is correct for single-digit days with
    optional single-letter suffixes (e.g. "3" < "3b" < "4").
    """
    info = []
    for day_dir in sorted(week_dir.glob("day*")):
        readme = day_dir / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        match = DAY_TITLE_PATTERN.match(first_line)
        if match:
            info.append({"num": match.group(1), "title": match.group(2).strip()})
    return sorted(info, key=lambda d: d["num"])


def _rewrite_day_image_links(text: str) -> str:
    """Rewrite image links in a day README (located at week/day/) to be relative
    to the week directory, where the generated HTML page is emitted.

    A link with N leading ``../`` segments is shifted up one level (N-1
    segments): ``images/`` and ``../images/`` both resolve to the week's own
    ``images/`` folder, while ``../../images/`` (shared daily images) resolves
    to ``../images/`` from the week page. The previous pattern only matched
    zero or one ``../`` segment, leaving ``../../images/`` untouched and
    causing 404s under the ``/ai-infra-notes/`` subpath deployment.
    """
    def repl(match: "re.Match[str]") -> str:
        prefix = match.group(1)
        depth = len(prefix) // 3  # number of "../" segments
        new_prefix = "../" * (depth - 1) if depth >= 1 else ""
        return "](" + new_prefix + "images/"
    return re.sub(r"\]\(((?:\.\./)*)images/", repl, text)


def load_overview_and_days(week_dir: Path):
    """Load overview from weekN/README.md and per-day markdown from weekN/dayN/README.md.

    Returns (overview_text, days) where days is a list of
    {"num": int, "title": str, "markdown": str} sorted by day number.
    """
    readme_path = week_dir / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Week README not found: {readme_path}")
    overview = readme_path.read_text(encoding="utf-8")
    overview = re.sub(r"\]\((?:\.\./)?images/", "](images/", overview)

    days = []
    for day_dir in sorted(week_dir.glob("day*")):
        readme = day_dir / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8")
        text = _rewrite_day_image_links(text)
        first_line = text.lstrip().splitlines()[0] if text.strip() else ""
        match = DAY_TITLE_PATTERN.match(first_line)
        if not match:
            raise ValueError(f"Cannot parse Day title from first line of {readme}: {first_line!r}")
        days.append({
            "num": match.group(1),
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
                r"^aiinfra/topics/([a-zA-Z0-9_-]+)/(.+)\.html$",
                r"\1/\2",
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
# Shared page template (VitePress look: light theme, navbar, back-nav,
# optional pill bar / eyebrow, right-side outline, prev/next pager).
# Used for every generated content page: weeks, topics, papers, plan,
# standalone solution/article pages.
# ---------------------------------------------------------------------------

def solution_page_template(
    title: str,
    markdown: str,
    *,
    back_link: Optional[tuple] = None,
    root_prefix: str = "",
    page_title: Optional[str] = None,
    eyebrow: str = "",
    subtitle: str = "",
    day_pills: Optional[list] = None,
    prev_link: Optional[tuple] = None,
    next_link: Optional[tuple] = None,
    extra_scripts: str = "",
) -> str:
    """Generate a content page in the VitePress style, mirroring the leetgpu
    GitHub Pages layout: top navbar with appearance switch, optional sticky
    day-pill bar, back-nav above the H1, right-side outline (本页目录), code
    blocks with line numbers, and a prev/next pager."""
    escaped_markdown = escape_for_template_string(markdown)
    escaped_title = html.escape(title, quote=True)
    if page_title is None:
        page_title = title

    back_nav_html = ""
    if back_link:
        back_href, back_label = back_link
        back_nav_html = (
            f'<nav class="back-nav"><a href="{back_href}">'
            f'← {html.escape(back_label, quote=True)}</a></nav>'
        )

    eyebrow_html = f'<div class="vp-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = (
        f'<p class="vp-doc-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    )

    body_class = "vp-page"
    pills_html = ""
    if day_pills:
        body_class += " has-pill-bar"
        items = []
        for pill in day_pills:
            active_cls = " active" if pill.get("active") else ""
            items.append(
                f'<a class="vp-pill{active_cls}" href="{pill["href"]}">{pill["label"]}</a>'
            )
        pills_html = (
            '<div class="vp-pill-bar"><div class="vp-pill-bar-inner">'
            + "".join(items)
            + "</div></div>"
        )

    def _pager(link, cls, label):
        if not link:
            return "<span></span>"
        href, text = link
        return (
            f'<a class="pager-link {cls}" href="{href}">'
            f'<span class="desc">{label}</span>'
            f'<span class="title">{html.escape(text, quote=True)}</span></a>'
        )

    prev_next_html = ""
    if prev_link or next_link:
        prev_next_html = (
            '<div class="prev-next">'
            + _pager(prev_link, "prev", "上一篇")
            + _pager(next_link, "next", "下一篇")
            + "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title}</title>
    <script>
    (function() {{
        try {{
            var t = localStorage.getItem('vp-theme') || 'auto';
            var dark = t === 'dark' ||
                (t !== 'light' && window.matchMedia('(prefers-color-scheme: dark)').matches);
            if (dark) document.documentElement.classList.add('dark');
        }} catch (e) {{}}
    }})();
    </script>
    <link rel="stylesheet" href="{root_prefix}css/vp-solution.css?v=2">
    <!-- Marked.js for Markdown rendering -->
    <script src="{root_prefix}js/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="{root_prefix}js/markdown-math.js"></script>
    <!-- Prism.js for syntax highlighting (token colors come from vp-solution.css) -->
    <script src="{root_prefix}js/prism.min.js"></script>
    <script src="{root_prefix}js/prism-c.min.js"></script>
    <script src="{root_prefix}js/prism-cpp.min.js"></script>
    <script>Prism.languages.cuda=Prism.languages.extend("c",{{builtin:/\\b(?:__global__|__device__|__host__|__shared__|__constant__|__managed__|__restrict__|__syncthreads|__threadfence|__threadfence_block|blockIdx|threadIdx|blockDim|gridDim|warpSize)\\b/}});</script>
    <script src="{root_prefix}js/prism-bash.min.js"></script>
    <script src="{root_prefix}js/prism-python.min.js"></script>
</head>
<body class="{body_class}">
    <header class="vp-navbar">
        <div class="vp-navbar-inner">
            <a class="vp-brand" href="{root_prefix}index.html">AI Infra <span>Notes</span></a>
            <div class="vp-navbar-spacer"></div>
            <nav class="vp-menu">
                <a href="{root_prefix}plan.html">10 周计划</a>
                <a href="{root_prefix}index.html#topics">专题笔记</a>
                <a href="{root_prefix}paper/index.html">论文精读</a>
                <a href="{GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
            </nav>
            <div class="vp-appearance">
                <button class="vp-switch" type="button" aria-label="切换深色/浅色外观">
                    <span class="check">
                        <span class="icon sun"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg></span>
                        <span class="icon moon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg></span>
                    </span>
                </button>
            </div>
        </div>
    </header>

    <div class="vp-main">
        {pills_html}
        <div class="vp-doc-wrap">
            <div class="vp-container">
                <div class="vp-content">
                    <div class="vp-content-container">
                        {back_nav_html}
                        <main class="main">
                            <article class="vp-doc">
                                {eyebrow_html}
                                <h1>{escaped_title}</h1>
                                {subtitle_html}
                                <div id="doc-content"></div>
                            </article>
                        </main>
                        <footer class="vp-doc-footer">{prev_next_html}</footer>
                    </div>
                </div>
                <div class="vp-aside">
                    <div class="vp-aside-container">
                        <div class="vp-aside-content">
                            <nav class="vp-outline" aria-labelledby="outline-title">
                                <div class="outline-title" id="outline-title">本页目录</div>
                                <div class="outline-content">
                                    <div class="outline-marker"></div>
                                    <ul class="outline-root" id="outline-root"></ul>
                                </div>
                            </nav>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="vp-footer">
        <span>AI Infra Notes · 由 <a href="{GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer">GitHub</a> 驱动 · Deployed on GitHub Pages</span>
    </footer>

    <button class="vp-back-to-top" aria-label="回到顶部">↑</button>

    <script src="{root_prefix}js/vp-solution.js?v=2"></script>
    <script>
        const pageMarkdown = `{escaped_markdown}`;

        try {{
            if (typeof marked === 'undefined' || !window.VPPage) {{
                throw new Error('页面脚本加载失败，请检查 js/marked.min.js 与 js/vp-solution.js 是否存在。');
            }}
            VPPage.render(pageMarkdown);
        }} catch (err) {{
            document.getElementById('doc-content').innerHTML = '<div style="padding: 20px; color: #b8272c; background: rgba(184,39,44,.08); border: 1px solid rgba(184,39,44,.3); border-radius: 8px;">' +
                '<h2>⚠️ 页面渲染失败</h2>' +
                '<p>' + err.message + '</p>' +
                '<p>请打开浏览器控制台查看详细错误。</p>' +
                '</div>';
            console.error('Markdown render error:', err);
        }}
    </script>
    {extra_scripts}
</body>
</html>
"""


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
