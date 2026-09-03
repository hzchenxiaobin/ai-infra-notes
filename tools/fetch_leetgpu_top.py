#!/usr/bin/env python3
"""Fetch the fastest LeetGPU solution for a given challenge.

Usage:
    # API mode (recommended): needs a LeetGPU auth token
    export LEETGPU_TOKEN="your_token_from_browser"
    python3 tools/fetch_leetgpu_top.py <slug> --out leetgpu/weekN/dayM/

    # Browser mode (fallback): uses Playwright with your local browser cookies
    python3 tools/fetch_leetgpu_top.py <slug> --out leetgpu/weekN/dayM/ --browser

Requires (API mode):
    pip install requests

Requires (browser mode):
    pip install playwright
    playwright install chromium

Authentication:
    LeetGPU's /api/v1/submissions and /api/v1/challenges/{id}/leaderboard endpoints
    require an Authorization token. Get it from the browser DevTools:
    1. Open https://leetgpu.com/challenges/<slug> and go to Solutions.
    2. F12 -> Network -> refresh.
    3. Find a request to api.leetgpu.com, copy the "Authorization" header
       (usually "Bearer eyJ...").
    4. Set it as LEETGPU_TOKEN, or pass --token "Bearer eyJ...".

For browser mode, Playwright will launch Chromium and use your existing
profile/session if you run it on your own machine while logged in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


API_BASE = "https://api.leetgpu.com"
WEB_BASE = "https://leetgpu.com"


def _parse_runtime(text: str) -> float:
    """Parse a runtime string like '0.0423ms' into milliseconds."""
    if not text:
        return float("inf")
    match = re.search(r"([0-9]*\.?[0-9]+)\s*(ms|us|s)?", str(text))
    if not match:
        return float("inf")
    value = float(match.group(1))
    unit = (match.group(2) or "ms").lower()
    if unit == "us":
        return value / 1000.0
    if unit == "s":
        return value * 1000.0
    return value


def _format_runtime(ms: float) -> str:
    if ms < 1.0:
        return f"{ms * 1000:.2f}us"
    return f"{ms:.4f}ms"


def _api_headers(token: Optional[str]) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = token if token.startswith("Bearer ") else f"Bearer {token}"
    return headers


def fetch_challenge_id_by_slug(slug: str) -> int:
    """Map challenge slug to its numeric id using the public challenges list."""
    if requests is None:
        raise RuntimeError("requests is required. Install: pip install requests")
    resp = requests.get(f"{API_BASE}/api/v1/challenges", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    for ch in data.get("challenges", []):
        title = ch.get("title", "")
        # Derive slug the same way LeetGPU does: lower, replace spaces with -, remove parens.
        derived = re.sub(r"[^\w\s-]", "", title.lower()).strip().replace(" ", "-")
        # Some slugs keep numbers as-is.
        if derived == slug or ch.get("slug") == slug or ch.get("id") == slug:
            return ch["id"]
    raise ValueError(f"Could not find challenge id for slug '{slug}'")


def fetch_leaderboard(challenge_id: int, token: Optional[str], language: str = "cuda", accelerator: str = "H100") -> list[dict]:
    """Fetch the challenge leaderboard for a given language and accelerator."""
    if requests is None:
        raise RuntimeError("requests is required. Install: pip install requests")
    url = f"{API_BASE}/api/v1/challenges/{challenge_id}/leaderboard"
    params = {"language": language, "accelerator": accelerator}
    resp = requests.get(url, params=params, headers=_api_headers(token), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("leaderboard", data if isinstance(data, list) else [])


def fetch_submission_code(submission_id: str, token: Optional[str]) -> str:
    """Fetch the source code for a submission id."""
    if requests is None:
        raise RuntimeError("requests is required. Install: pip install requests")
    url = f"{API_BASE}/api/v1/submissions/{submission_id}/code"
    resp = requests.get(url, headers=_api_headers(token), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Common field names
    for key in ("code", "source", "content", "solution"):
        if key in data and isinstance(data[key], str):
            return data[key]
    return data.get("data", "") if isinstance(data.get("data"), str) else json.dumps(data, indent=2)


def fetch_top_via_api(slug: str, token: Optional[str], top: int = 1, language: str = "cuda", accelerator: str = "H100") -> dict:
    """Fetch the top-ranked solution via LeetGPU API."""
    challenge_id = fetch_challenge_id_by_slug(slug)
    leaderboard = fetch_leaderboard(challenge_id, token, language, accelerator)
    if not leaderboard:
        raise RuntimeError(f"Leaderboard for '{slug}' is empty. Check token / language / accelerator.")

    # Sort by runtime (ascending). LeetGPU usually returns it sorted, but be safe.
    def runtime_key(entry: dict) -> float:
        for key in ("runtime", "runtime_ms", "time", "time_ms", "duration", "duration_ms"):
            if key in entry:
                return _parse_runtime(entry[key])
        return float("inf")

    leaderboard.sort(key=runtime_key)
    chosen = leaderboard[max(0, top - 1)]
    submission_id = chosen.get("id") or chosen.get("submission_id") or chosen.get("submissionId")
    if not submission_id:
        raise RuntimeError(f"Leaderboard entry has no submission id: {chosen}")

    code = fetch_submission_code(str(submission_id), token)
    runtime_ms = runtime_key(chosen)
    return {
        "username": chosen.get("username") or chosen.get("user", {}).get("username"),
        "runtime_text": chosen.get("runtime") or _format_runtime(runtime_ms),
        "runtime_ms": runtime_ms,
        "code": code,
        "source_url": f"{WEB_BASE}/challenges/{slug}",
        "accelerator": accelerator,
        "language": language,
    }


def fetch_top_via_playwright(slug: str, top: int = 1, headless: bool = True) -> dict:
    """Use Playwright to extract the fastest solution code from the LeetGPU UI."""
    if sync_playwright is None:
        raise RuntimeError(
            "playwright is required for browser mode. Install: pip install playwright && playwright install chromium"
        )

    url = f"{WEB_BASE}/challenges/{slug}"
    result = {
        "username": None,
        "runtime_text": None,
        "runtime_ms": None,
        "code": None,
        "source_url": url,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        try:
            page.goto(url, wait_until="networkidle", timeout=60000)
            for locator in [
                page.locator("text=Solutions").first,
                page.locator("button:has-text('Solutions')").first,
                page.locator("a:has-text('Solutions')").first,
            ]:
                try:
                    if locator.is_visible(timeout=3000):
                        locator.click()
                        page.wait_for_timeout(1000)
                        break
                except Exception:
                    pass

            page.wait_for_selector("text=/Runtime:", timeout=30000)

            cards = page.locator("text=Runtime:").locator("xpath=../..").all()
            if not cards:
                cards = page.locator("*:has-text('Runtime:')").all()

            entries = []
            for card in cards:
                text = card.inner_text()
                runtime_match = re.search(r"Runtime:\s*([0-9]*\.?[0-9]+\s*(ms|us|s)?)", text)
                if not runtime_match:
                    continue
                runtime_text = runtime_match.group(1)
                runtime_ms = _parse_runtime(runtime_text)
                username = None
                user_match = re.search(r"\n([^\n]+)\nRuntime:", text)
                if user_match:
                    username = user_match.group(1).strip()
                entries.append(
                    {
                        "card": card,
                        "username": username,
                        "runtime_text": runtime_text,
                        "runtime_ms": runtime_ms,
                    }
                )

            if not entries:
                raise RuntimeError("Could not find any solution cards with Runtime.")

            entries.sort(key=lambda x: x["runtime_ms"])
            chosen = entries[max(0, top - 1)]
            result["username"] = chosen["username"]
            result["runtime_text"] = chosen["runtime_text"]
            result["runtime_ms"] = chosen["runtime_ms"]

            view_code = chosen["card"].locator("button:has-text('View Code')").first
            if not view_code.is_visible(timeout=3000):
                buttons = page.locator("button:has-text('View Code')").all()
                if top - 1 < len(buttons):
                    view_code = buttons[top - 1]
                else:
                    view_code = buttons[-1]
            view_code.click()

            code = ""
            for selector in [
                "pre[class*='code']",
                "[class*='monaco']",
                ".cm-content",
                "textarea[readonly]",
                "code",
            ]:
                try:
                    page.wait_for_selector(selector, timeout=5000)
                    el = page.locator(selector).first
                    code = el.inner_text()
                    if code.strip():
                        break
                except Exception:
                    continue

            result["code"] = code
        finally:
            context.close()
            browser.close()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the fastest LeetGPU solution.")
    parser.add_argument("slug", help="Challenge slug, e.g. softmax-attention")
    parser.add_argument("--top", type=int, default=1, help="Which rank to fetch (1=fastest)")
    parser.add_argument("--out", type=Path, default=Path("."), help="Output directory or file path")
    parser.add_argument("--token", help="LeetGPU Authorization token (or set LEETGPU_TOKEN env)")
    parser.add_argument("--language", default="cuda", help="Language filter for leaderboard")
    parser.add_argument("--accelerator", default="H100", help="Accelerator filter for leaderboard")
    parser.add_argument("--browser", action="store_true", help="Use Playwright browser mode instead of API")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run browser in headless mode")
    args = parser.parse_args()

    token = args.token or os.environ.get("LEETGPU_TOKEN")
    out_path = args.out
    if out_path.is_dir():
        out_path = out_path / f"best-solution-{args.slug}.cu"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.browser:
        print(f"Opening browser for: {args.slug}", file=sys.stderr)
        result = fetch_top_via_playwright(args.slug, args.top, args.headless)
    else:
        if not token:
            print(
                "Error: LeetGPU API requires authentication.\n"
                "Set LEETGPU_TOKEN env var or pass --token.\n"
                "Or use --browser for Playwright mode.",
                file=sys.stderr,
            )
            return 1
        print(f"Fetching API for: {args.slug}", file=sys.stderr)
        result = fetch_top_via_api(args.slug, token, args.top, args.language, args.accelerator)

    header_lines = [
        f"// best-solution-{args.slug}.cu -- LeetGPU top {args.top} solution",
    ]
    if result.get("username") and result.get("runtime_text"):
        header_lines.append(f"// Author: {result['username']} / Runtime: {result['runtime_text']} / {result.get('accelerator', 'H100')}")
    header_lines.append(f"// Source: {result['source_url']}")
    header_lines.append("")

    out_path.write_text("\n".join(header_lines) + (result["code"] or "") + "\n", encoding="utf-8")
    print(f"Saved {out_path}")
    print(f"Runtime: {result['runtime_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
