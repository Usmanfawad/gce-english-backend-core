from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger
from playwright.sync_api import sync_playwright

from app.config.settings import settings
from openai import OpenAI


@dataclass
class VisualSnapshot:
    url: str
    title: str
    host: str
    screenshot_path: Optional[Path]
    extracted_text_path: Optional[Path]
    created_at: datetime


def _hash(input_str: str) -> str:
    return hashlib.sha1(input_str.encode("utf-8")).hexdigest()[:16]


def _ensure_session_dir(key: str) -> Path:
    dir_path = settings.visual_output_dir / key
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _validate_url(url: str) -> bool:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=8)
        if resp.status_code >= 400:
            return False
        ctype = resp.headers.get("content-type", "")
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            return False
        return True
    except Exception:
        return False


def _fetch_html(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code >= 400:
            return None
        # limit size
        if len(resp.content) > 3_000_000:
            return None
        return resp.text
    except Exception:
        return None


def _readable_text(html: str) -> Tuple[str, str]:
    """Return (title, readable_text) using basic heuristics via BeautifulSoup."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")[:200]
    # Prefer <article>, then <main>, else longest <div>
    candidates = []
    for sel in ["article", "main"]:
        node = soup.select_one(sel)
        if node:
            text = node.get_text(separator="\n", strip=True)
            candidates.append(text)
    if not candidates:
        longest_div = max((d.get_text(separator="\n", strip=True) for d in soup.find_all("div")), key=lambda t: len(t), default="")
        candidates.append(longest_div)
    text = max(candidates, key=lambda t: len(t), default="")
    # Fallback to body text
    if len(text) < 200 and soup.body:
        text = soup.body.get_text(separator="\n", strip=True)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text[:12000]


def _clear_fixed_overlays(page: Any) -> None:
    """Hard-remove stubborn fixed/sticky overlays that block the viewport."""
    try:
        page.evaluate(
            """
            () => {
                const vw = window.innerWidth || 1280;
                const vh = window.innerHeight || 720;
                const selectors = [
                    '[role="dialog"]',
                    '[aria-modal="true"]',
                    '.modal',
                    '.modal-overlay',
                    '.modal-backdrop',
                    '.popup',
                    '.popup-overlay',
                    '.lightbox',
                    '.newsletter-modal',
                    '.interstitial',
                    '.signup-modal',
                    '.subscription-overlay',
                    '.overlay-mask',
                    '.cboxOverlay',
                ];
                for (const sel of selectors) {
                    document.querySelectorAll(sel).forEach(el => el.remove());
                }
                const candidates = Array.from(document.querySelectorAll('body *')).filter(el => {
                    const style = window.getComputedStyle(el);
                    return (
                        (style.position === 'fixed' || style.position === 'sticky') &&
                        (parseInt(style.zIndex || '0', 10) >= 800) &&
                        el.offsetWidth >= vw * 0.35 &&
                        el.offsetHeight >= vh * 0.35
                    );
                });
                candidates.forEach(el => el.remove());
            }
            """
        )
    except Exception:
        pass


def _dismiss_modal_overlays(page: Any, *, attempts: int = 3) -> None:
    """
    Best-effort dismissal of pop-ups or modals (e.g., close icons, dismiss buttons)
    so screenshots capture the main visual. Safe to call even if nothing is present.
    """
    close_selectors = [
        'button:has-text("Close")',
        'button:has-text("CLOSE")',
        'button:has-text("close")',
        'button:has-text("Dismiss")',
        'button:has-text("No Thanks")',
        'button:has-text("×")',
        'span:has-text("×")',
        'div:has-text("×")',
        '[aria-label="Close"]',
        '[aria-label="close"]',
        '[aria-label*="Close"]',
        '[aria-label*="close"]',
        '[title*="Close"]',
        '[title*="close"]',
        '.close-button',
        '.close-btn',
        '.modal-close',
        '.popup-close',
        '.close-icon',
        '.icon-close',
        '.btn-close',
        'button[aria-label*="dismiss"]',
        '[role="button"][aria-label*="close"]',
        '[role="button"][title*="close"]',
        '[data-action="close"]',
        '[data-testid*="close"]',
        '[class*="close"][role="button"]',
    ]
    for _ in range(attempts):
        dismissed = False
        for selector in close_selectors:
            try:
                locator = page.locator(selector).first
                if locator.is_visible(timeout=800):
                    locator.click(timeout=1200)
                    logger.info(f"Dismissed modal/popup via selector: {selector}")
                    page.wait_for_timeout(600)
                    dismissed = True
                    break
            except Exception:
                continue
        if not dismissed:
            break
    _clear_fixed_overlays(page)


def _extract_visible_viewport_content(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract title and visible viewport content from a URL using Playwright.
    This matches what's actually shown in the screenshot (viewport only, scrolled 200px down)."""
    try:
        def _run():
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1920, "height": 1080})
                page.goto(url, wait_until="load", timeout=30000)
                page.wait_for_load_state("networkidle", timeout=15000)
                
                # Dismiss cookie banners (same logic as screenshot)
                cookie_selectors = [
                    'button:has-text("Accept All Cookies")', 'button:has-text("Accept All")',
                    'button:has-text("Accept Cookies")', 'button:has-text("I Accept")',
                    'button:has-text("Accept")', 'button:has-text("Agree")',
                    'button:has-text("I Agree")', 'a:has-text("Accept All Cookies")',
                    'a:has-text("Accept All")', 'a:has-text("Accept Cookies")',
                    '#accept-all-cookies', '#acceptAllCookies', '#cookie-accept',
                    '#accept-cookies', '.accept-all-cookies', '.acceptAllCookies',
                    '.cookie-accept', '.accept-cookies',
                    '[id*="accept"][id*="cookie"]', '[class*="accept"][class*="cookie"]',
                    '[data-testid*="accept"]', '[aria-label*="Accept"]', '[aria-label*="accept"]',
                ]
                for selector in cookie_selectors:
                    try:
                        button = page.locator(selector).first
                        if button.is_visible(timeout=2000):
                            button.click(timeout=3000)
                            page.wait_for_timeout(1000)
                            break
                    except Exception:
                        continue

                _dismiss_modal_overlays(page)
                
                # Scroll down 200px (same as screenshot)
                page.evaluate("window.scrollTo(0, 200)")
                page.wait_for_timeout(500)
                
                # Extract visible content from viewport
                visible_content = page.evaluate("""
                    () => {
                        const viewportHeight = window.innerHeight;
                        const viewportWidth = window.innerWidth;
                        const elements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, div, span, a, li, td, th, article, main, section');
                        let textParts = [];
                        let title = '';
                        
                        // Get title
                        if (document.title) title = document.title;
                        else {
                            const h1 = document.querySelector('h1');
                            if (h1) {
                                const rect = h1.getBoundingClientRect();
                                if (rect.top >= 0 && rect.top < viewportHeight && rect.left >= 0 && rect.left < viewportWidth) {
                                    title = h1.innerText.trim();
                                }
                            }
                        }
                        
                        // Collect visible text elements
                        const seen = new Set();
                        for (const el of elements) {
                            const rect = el.getBoundingClientRect();
                            // Check if element is visible in viewport (after scrolling 200px)
                            if (rect.top >= 0 && rect.top < viewportHeight && 
                                rect.left >= 0 && rect.left < viewportWidth &&
                                rect.width > 0 && rect.height > 0) {
                                const text = el.innerText.trim();
                                if (text && text.length > 5 && !seen.has(text)) {
                                    // Avoid duplicates by checking if parent is already included
                                    let isChild = false;
                                    for (const parent of elements) {
                                        if (parent !== el && parent.contains(el)) {
                                            const parentRect = parent.getBoundingClientRect();
                                            if (parentRect.top >= 0 && parentRect.top < viewportHeight) {
                                                isChild = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (!isChild) {
                                        seen.add(text);
                                        textParts.push(text);
                                    }
                                }
                            }
                        }
                        return { title: title, text: textParts.join('\\n\\n').substring(0, 5000) };
                    }
                """)
                
                browser.close()
                return visible_content.get("title", ""), visible_content.get("text", "")
        with ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(_run).result()
            return result
    except Exception as e:
        logger.warning(f"Failed to extract visible viewport content from {url}: {e}")
        return None, None


def _screenshot_url(url: str, out_path: Path) -> bool:
    try:
        def _run():
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # Set desktop viewport size (1920x1080 for standard desktop)
                page = browser.new_page(viewport={"width": 1920, "height": 1080})
                page.goto(url, wait_until="load", timeout=30000)
                page.wait_for_load_state("networkidle", timeout=15000)
                
                # Try to dismiss cookie consent banners
                cookie_selectors = [
                    # Text-based selectors (case-insensitive)
                    'button:has-text("Accept All Cookies")',
                    'button:has-text("Accept All")',
                    'button:has-text("Accept Cookies")',
                    'button:has-text("I Accept")',
                    'button:has-text("Accept")',
                    'button:has-text("Agree")',
                    'button:has-text("I Agree")',
                    'a:has-text("Accept All Cookies")',
                    'a:has-text("Accept All")',
                    'a:has-text("Accept Cookies")',
                    # ID and class-based selectors
                    '#accept-all-cookies',
                    '#acceptAllCookies',
                    '#cookie-accept',
                    '#accept-cookies',
                    '.accept-all-cookies',
                    '.acceptAllCookies',
                    '.cookie-accept',
                    '.accept-cookies',
                    '[id*="accept"][id*="cookie"]',
                    '[class*="accept"][class*="cookie"]',
                    '[data-testid*="accept"]',
                    # Common cookie banner button patterns
                    '[aria-label*="Accept"]',
                    '[aria-label*="accept"]',
                ]
                
                cookie_dismissed = False
                for selector in cookie_selectors:
                    try:
                        # Try to find and click the button
                        button = page.locator(selector).first
                        if button.is_visible(timeout=2000):
                            button.click(timeout=3000)
                            logger.info(f"Clicked cookie consent button: {selector} for {url}")
                            # Wait a bit for the banner to disappear
                            page.wait_for_timeout(1000)
                            cookie_dismissed = True
                            break
                    except Exception:
                        # Selector didn't match or click failed, try next
                        continue
                
                if cookie_dismissed:
                    # Wait a bit more for any animations/transitions
                    page.wait_for_timeout(500)
                    # Re-wait for network idle after dismissing banner
                    try:
                        page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:
                        pass  # Continue even if networkidle times out

                _dismiss_modal_overlays(page)
                
                # Scroll down a bit to avoid header/navigation bars (scroll 200px)
                page.evaluate("window.scrollTo(0, 200)")
                page.wait_for_timeout(500)  # Wait for scroll to complete
                
                # Capture only viewport (visible area), not full page
                page.screenshot(path=str(out_path), full_page=False)
                browser.close()
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(_run).result()
        return out_path.exists()
    except Exception as e:
        logger.warning(f"Screenshot failed for {url}: {e}")
        return False


def _build_description(title: str, text: str) -> str:
    lines = []
    heading = title or "Visual: Informational Web Page"
    lines.append(f"Heading: {heading}")
    # Extract 2–3 short 'callouts' (first sentences)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    callouts = [s.strip() for s in sentences if 30 <= len(s) <= 140][:3]
    if callouts:
        lines.append("Callouts:")
        for c in callouts[:3]:
            lines.append(f"- {c}")
    # Short blurbs from paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 0]
    blurbs = [p[:140] + ("..." if len(p) > 140 else "") for p in paragraphs[:3]]
    if blurbs:
        lines.append("Blurbs:")
        for b in blurbs:
            lines.append(f"- {b}")
    lines.append("Layout cues: banner at top, body text below, possible side panel or callout box.")
    return "\n".join(lines)


def _openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=settings.openai_api_key)


def _llm_candidate_urls(topics: Iterable[str], paper_format: str, section: Optional[str], paper_name: str) -> List[str]:
    client = _openai_client()
    topic_str = ", ".join(topics) if topics else "general"
    section_str = section or "full"
    prompt = (
        "You are sourcing PUBLIC, HEADLESS-BROWSER-FRIENDLY HTML pages that can act as visual stimuli for a "
        "GCE O-Level English examination paper. I will screenshot the page exactly as rendered on a desktop monitor "
        "and embed it beside the question.\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. Output ONLY a JSON array of fully qualified HTTPS URLs (no commentary, no code fences).\n"
        "2. Each URL must render a visually rich layout within the first viewport: hero banner, callouts, promo panels, "
        "infographic blocks, or clearly structured headings + subtext.\n"
        "3. Must be freely accessible (no login, no paywall, no heavy cookie wall) and not rely on infinite scrolling.\n"
        "4. PREFERRED CONTENT TYPES for O-Level English visual text comprehension:\n"
        "   - Community event announcements (festivals, exhibitions, charity events)\n"
        "   - Educational program pages (workshops, courses, summer camps)\n"
        "   - Environmental/sustainability campaigns\n"
        "   - Health and wellness initiatives\n"
        "   - Youth programs and school activities\n"
        "   - Cultural and arts events\n"
        "   - Library/museum exhibitions and programs\n"
        "   - Volunteer opportunities\n"
        "   - Sports and recreation programs\n"
        "5. HARD-BLOCK these categories/domains:\n"
        "   - Social media (facebook, instagram, tiktok, youtube, twitter/x, reddit)\n"
        "   - Q&A forums (quora / stackoverflow)\n"
        "   - News paywalls (nytimes, washingtonpost, cnn)\n"
        "   - Government or UN/UNESCO domains\n"
        "   - Exam-prep sites, PDF-only resources\n"
        "   - VISA/IMMIGRATION related content (visa applications, immigration services, work permits, etc.)\n"
        "   - Commercial advertisements (mock ads, product ads, services selling)\n"
        "   - Sites with heavy cloudflare protection\n"
        "6. Avoid generic homepages—return specific landing pages, event/announcement pages, or campaign pages.\n"
        "7. At least 5 and up to 8 unique URLs, all different hosts.\n\n"
        f"CONTEXT:\n"
        f"- Paper format: {paper_format}\n"
        f"- Section: {section_str}\n"
        f"- Topics to reflect: {topic_str}\n"
        f"- Desired tone: modern Singapore GCE O-Level English exam stimulus.\n\n"
        "Respond with JSON only, e.g. [\"https://example.com/page\", \"https://travel.org/event\"]."
    )
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "You return clean JSON only, no commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    logger.info(f"LLM URL candidates raw: {content[:2000]}{'...[truncated]' if len(content) > 2000 else ''}")
    # Strip code fences if present
    fenced = content.strip()
    if fenced.startswith("```"):
        fenced = re.sub(r"^```(?:json)?\s*", "", fenced)
        fenced = re.sub(r"\s*```$", "", fenced)
    try:
        data = json.loads(fenced)
        urls = [u for u in data if isinstance(u, str)]
        logger.info(f"LLM provided URLs: {urls}")
        return urls[:5]
    except Exception:
        # Fallback: extract URLs by regex
        urls = re.findall(r"https?://[^\s\"'\]]+", content)
        logger.info(f"Extracted URLs via regex: {urls}")
        return urls[:5]


def _tavily_urls(topics: Optional[List[str]] = None) -> List[str]:
    """Search Tavily for URLs related to topics, excluding problematic domains. Returns only URLs."""
    if not settings.tavily_api_key:
        return []

    query_parts = [
        "visual stimulus html page",
        "travel event community campaign poster",
    ]
    if topics:
        query_parts.append(", ".join(topics))
    query = " ".join(query_parts)

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": 20,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        resp = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        urls = [item.get("url") for item in results if item.get("url")]
    except Exception as exc:
        logger.warning(f"Tavily search failed: {exc}")
        return []

    excluded_domains = [
        "quora.com",
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "tiktok.com",
        "twitter.com",
        "x.com",
        "reddit.com",
        "nytimes.com",
        "washingtonpost.com",
        "cnn.com",
        "exam-papers.com",
        "pastpapers.com",
        "exam-mate.com",
        "cambridge.org",
        "cie.org.uk",
        "unacademy.com",
        "unesco.org",
        "un.org",
        "scribd.com",
        "pinterest.com",
        "postermywall.com",
        "arxiv.org",
        "museumofscience.org",
        "cookridgeprimary.co.uk"
        "anaheim.net"
        # Immigration/visa related domains
        "ica.gov.sg",
        "immigration",
        "visa",
        "immi.gov",
        "uscis.gov",
        "ukvi",
        "homeoffice.gov",
    ]

    # Keywords to filter out from URLs (immigration/visa/ads/government related)
    excluded_keywords = [
        "visa",
        "immigration",
        "immi",
        "passport",
        "citizenship",
        "residency",
        "work-permit",
        "green-card",
        "pr-application",
        "migrate",
        "ads",
        "advertisement",
        "sponsor",
        "affiliate",
        # Government/municipal website patterns
        "cityof",
        "city-of",
        "countyof",
        "county-of",
        "stateof",
        "state-of",
        "townof",
        "town-of",
        "government",
        "municipal",
        "council",
        "public-services",
        "civic",
    ]

    filtered: List[str] = []
    for url in urls:
        lower = url.lower()
        if lower.endswith(".pdf"):
            continue
        if any(dom in lower for dom in excluded_domains):
            continue
        if any(kw in lower for kw in excluded_keywords):
            logger.debug(f"Filtered out URL with excluded keyword: {url}")
            continue
        if ".gov" in lower:
            continue
        # Filter government-style .net/.org domains
        if "city" in lower and any(ext in lower for ext in [".net", ".org", ".us"]):
            logger.debug(f"Filtered out government-style URL: {url}")
            continue
        filtered.append(url)

    seen: set[str] = set()
    unique_urls: List[str] = []
    for url in filtered:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    logger.info(
        f"Tavily search completed | topics={topics} | urls_found={len(urls)} | "
        f"filtered={len(unique_urls)} | urls={unique_urls[:5]}{'...' if len(unique_urls) > 5 else ''}"
    )
    return unique_urls


def _openai_web_search_urls(topics: Optional[List[str]] = None) -> List[str]:
    """Search the web using OpenAI's web_search tool to find URLs for visual stimuli.
    Returns a list of URLs suitable for screenshots, excluding problematic domains."""
    if not settings.openai_api_key:
        return []
    unique_urls: List[str] = []

    try:
        client = _openai_client()

        # Build a detailed, constrained prompt that forces JSON output
        if topics:
            topic_str = ", ".join(topics)
            search_query = (
                "You are sourcing publicly accessible HTML web pages that contain rich visual stimulus content "
                "for GCE O-Level English examination questions.\n\n"
                f"TARGET TOPICS: {topic_str}\n\n"
                "REQUIREMENTS:\n"
                "1. The page must be publicly accessible (no login/paywall) and render in a headless Chromium browser.\n"
                "2. Content must feature clear visual layout elements visible within the initial viewport after a small scroll "
                "(hero banners, callouts, headings, infographics, promotional sections, etc.).\n"
                "3. PREFERRED CONTENT TYPES for O-Level English:\n"
                "   - Community event announcements (festivals, exhibitions, charity events)\n"
                "   - Educational program pages (workshops, courses, summer camps)\n"
                "   - Environmental/sustainability campaigns\n"
                "   - Health and wellness initiatives\n"
                "   - Youth programs and school activities\n"
                "   - Cultural and arts events\n"
                "   - Library/museum exhibitions and programs\n"
                "   - Volunteer opportunities\n"
                "   - Sports and recreation programs\n"
                "4. STRICTLY AVOID:\n"
                "   - Social media, Q&A forums, video-only sites, news paywalls\n"
                "   - Exam-prep sites, government PDFs\n"
                "   - VISA/IMMIGRATION content (visa applications, immigration services, work permits)\n"
                "   - Commercial advertisements and mock ads\n"
                "   - Domains: facebook.com, instagram.com, tiktok.com, youtube.com, quora.com, "
                "reddit.com, nytimes.com, washingtonpost.com, cnn.com, *.gov, cambridge.org, cie.org.uk, unesco.org, un.org\n"
                "5. The final screenshot should make sense on its own; therefore, URLs that are mostly text with no layout cues "
                "should be skipped.\n\n"
                "TASK:\n"
                "Use web_search to gather at least 12 strong candidate URLs that satisfy the above. "
                "Return ONLY a JSON array of fully-qualified https URLs with no commentary, e.g. "
                '["https://example.com/page1", "https://example.org/page2"].'
            )
        else:
            search_query = (
                "Find publicly accessible HTML pages with visually rich layouts (banners, callouts, headings) suitable for "
                "GCE O-Level English visual stimulus questions. Prefer community events, educational programs, environmental "
                "campaigns, cultural events, youth activities, library/museum programs, volunteer opportunities. "
                "AVOID social media, Q&A, paywalled news, government PDFs, exam-prep sites, AND visa/immigration content. "
                "Use web_search. Return ONLY a JSON array of https URLs like [\"https://example.com\", ...] with at least 12 entries."
            )

        response = client.responses.create(
            model=settings.openai_model,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=search_query,
        )

        raw_text = (getattr(response, "output_text", None) or "").strip()
        if not raw_text and hasattr(response, "output"):
            # Fallback: concatenate textual parts from response.output if output_text missing
            chunks = []
            for item in getattr(response, "output", []) or []:
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", "") in {"output_text", "text"} and hasattr(part, "text"):
                        chunks.append(part.text)
            raw_text = "\n".join(chunks).strip()

        logger.info(
            "OpenAI web search raw output",
            extra={"raw_preview": raw_text[:500] if raw_text else ""},
        )

        if not raw_text:
            return []

        # Strip code fences if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        urls: List[str] = []
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "urls" in parsed and isinstance(parsed["urls"], list):
                urls = [u for u in parsed["urls"] if isinstance(u, str)]
            elif isinstance(parsed, list):
                urls = [u for u in parsed if isinstance(u, str)]
        except Exception:
            urls = re.findall(r"https?://[^\s\"'\]]+", raw_text)

        # Filter out problematic domains and file types
        filtered_urls = []
        excluded_domains = [
            "quora.com",
            "facebook.com",
            "instagram.com",
            "youtube.com",
            "tiktok.com",
            "twitter.com",
            "x.com",
            "reddit.com",
            "nytimes.com",
            "washingtonpost.com",
            "cnn.com",
            "exam-papers.com",
            "pastpapers.com",
            "exam-mate.com",
            "cambridge.org",
            "cie.org.uk",
            "unacademy.com",
            "unesco.org",
            "un.org",
            "scribd.com",
            "pinterest.com",
            "postermywall.com",
            "arxiv.org",
            "museumofscience.org",
            "cookridgeprimary.co.uk",
            # Immigration/visa related domains
            "ica.gov.sg",
            "immigration",
            "visa",
            "immi.gov",
            "uscis.gov",
            "ukvi",
            "homeoffice.gov",
            "anaheim.net",
        ]

        # Keywords to filter out from URLs (immigration/visa/ads/government related)
        excluded_keywords = [
            "visa",
            "immigration",
            "immi",
            "passport",
            "citizenship",
            "residency",
            "work-permit",
            "green-card",
            "pr-application",
            "migrate",
            "ads",
            "advertisement",
            "sponsor",
            "affiliate",
            # Government/municipal website patterns
            "cityof",
            "city-of",
            "countyof",
            "county-of",
            "stateof",
            "state-of",
            "townof",
            "town-of",
            "government",
            "municipal",
            "council",
            "public-services",
            "civic",
        ]

        for url in urls:
            url_lower = url.lower()
            if url_lower.endswith(".pdf"):
                logger.debug(f"Filtered out PDF: {url}")
                continue
            if any(excluded in url_lower for excluded in excluded_domains):
                logger.debug(f"Filtered out excluded domain: {url}")
                continue
            if any(kw in url_lower for kw in excluded_keywords):
                logger.debug(f"Filtered out URL with excluded keyword: {url}")
                continue
            if ".gov" in url_lower:
                logger.debug(f"Filtered out government domain: {url}")
                continue
            # Filter government-style .net/.org domains
            if "city" in url_lower and any(ext in url_lower for ext in [".net", ".org", ".us"]):
                logger.debug(f"Filtered out government-style URL: {url}")
                continue
            filtered_urls.append(url)

        seen = set()
        unique_urls = []
        for url in filtered_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        logger.info(
            f"OpenAI web search completed | topics={topics} | urls_found={len(urls)} | "
            f"filtered={len(unique_urls)} | urls={unique_urls[:5]}{'...' if len(unique_urls) > 5 else ''}"
        )

        unique_urls.reverse()
        return unique_urls
    except Exception as e:
        logger.warning(f"OpenAI web search failed: {e}")
        return unique_urls

def get_visual(
    *,
    topics: Optional[Iterable[str]],
    paper_format: str,
    section: Optional[str],
    search_provider: str = "openai",  # Changed from "tavily" to "openai"
) -> Tuple[Optional[VisualSnapshot], Optional[str]]:
    """Return a snapshot and a textual description; if screenshot fails, snapshot may be None, but description may still be present."""
    topic_list = list(topics or [])
    def try_urls(candidate_urls: List[str]) -> Tuple[Optional[VisualSnapshot], Optional[str], bool]:
        last_desc: Optional[str] = None
        logger.info(f"Visual search starting | provider={search_provider} | urls={candidate_urls}")
        for url in candidate_urls:
            logger.info(f"Trying candidate URL: {url}")
            if not _validate_url(url):
                logger.info(f"URL failed validation: {url}")
                continue
            
            # Extract visible viewport content (matches what's in screenshot)
            visible_title, visible_text = _extract_visible_viewport_content(url)
            if not visible_text:
                # Fallback to HTML parsing if Playwright extraction fails
                html = _fetch_html(url)
                if html:
                    title, readable = _readable_text(html)
                    # Limit to approximate viewport content (first 3000 chars)
                    visible_text = readable[:3000]
                    visible_title = title
                else:
                    logger.info(f"Content extraction failed: {url}")
                    continue
            else:
                logger.info(f"Extracted visible viewport content | url={url} | text_len={len(visible_text)}")
            
            # Build description from visible content only (what's actually in screenshot)
            description = _build_description(visible_title or "Web Page", visible_text)
            last_desc = description
            
            # Attempt screenshot
            key = _hash(url + str(datetime.utcnow().date()))
            out_dir = _ensure_session_dir(key)
            shot = out_dir / "screenshot.png"
            text_out = out_dir / "extracted.txt"
            text_out.write_text(visible_text, encoding="utf-8")
            ok = _screenshot_url(url, shot)
            logger.info(
                f"Visual snapshot | url={url} | shot={ok} | visible_text_len={len(visible_text)} | "
                f"shot_path={shot}"
            )
            parsed = urlparse(url)
            snapshot = VisualSnapshot(
                url=url,
                title=visible_title or parsed.hostname or "Web Page",
                host=parsed.hostname or "",
                screenshot_path=shot if ok else None,
                extracted_text_path=text_out,
                created_at=datetime.utcnow(),
            )
            if ok:
                return snapshot, description, True
            # else continue to next candidate
            time.sleep(1.0)
        return None, last_desc, False

    urls: List[str] = []

    def _extend_unique(target: List[str], items: List[str]) -> None:
        seen_local = set(target)
        for item in items:
            if item not in seen_local:
                target.append(item)
                seen_local.add(item)

    topics_list = topic_list if topic_list else None

    if search_provider == "openai":
        _extend_unique(urls, _openai_web_search_urls(topics=topics_list))
        if not urls:
            _extend_unique(urls, _tavily_urls(topics=topics_list))
    elif search_provider == "tavily":
        _extend_unique(urls, _tavily_urls(topics=topics_list))
    elif search_provider == "hybrid":
        _extend_unique(urls, _openai_web_search_urls(topics=topics_list))
        _extend_unique(urls, _tavily_urls(topics=topics_list))

    # Fallback to LLM URL generation if previous searches didn't return URLs
    if not urls and search_provider in {"llm", "hybrid", "openai", "tavily"}:
        _extend_unique(
            urls,
            _llm_candidate_urls(topic_list, paper_format, section, paper_format.replace("_", " ")),
        )
    
    # Deduplicate
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]
    snapshot, description, success = try_urls(urls)
    if success:
        return snapshot, description

    # If hybrid and LLM was tried first, try OpenAI web search next
    if search_provider == "hybrid" and urls:
        openai_urls = _openai_web_search_urls(topics=topic_list if topic_list else None)
        # Dedup with previously tried
        openai_urls = [u for u in openai_urls if u not in urls]
        snapshot, description2, success2 = try_urls(openai_urls)
        if success2:
            return snapshot, description2
        description = description or description2

    logger.warning("All visual candidates failed; returning description-only")
    return None, description


