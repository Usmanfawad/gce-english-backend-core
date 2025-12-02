from __future__ import annotations

from pathlib import Path
import re
import html as _html
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
from loguru import logger
from xhtml2pdf import pisa

from app.config.settings import settings


def _env() -> Environment:
    loader = FileSystemLoader(str(settings.html_template_dir))
    return Environment(loader=loader, autoescape=select_autoescape(["html", "xml"]))


def _build_p1_section_a_html(content: str) -> Optional[str]:
    """
    Detect and render Paper 1 Section A numbered 12-line passage into a single table
    with the passage text on the left and answer blanks on the right (side-by-side).
    Expects lines like `1. ...` through `12. ...`.
    """
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    numbered = []
    for ln in lines:
        # Handle both single-digit (1. ) and double-digit (10. ) line numbers
        match = re.match(r'^(\d{1,2})\.\s+(.*)$', ln)
        if match:
            try:
                num = int(match.group(1))
                text = match.group(2).strip()
                numbered.append((num, text))
            except ValueError:
                continue
    nums = {n for n, _ in numbered}
    if not (len(numbered) >= 12 and all(i in nums for i in range(1, 13))):
        return None

    # Build single combined table: Line | Text | Answer
    rows_html = [
        "<tr>"
        "<th style='text-align:center; width:12mm; font-weight:bold;'>Line</th>"
        "<th style='text-align:left; font-weight:bold;'>Text</th>"
        "<th style='text-align:center; width:50mm; font-weight:bold;'>Correction</th>"
        "</tr>"
    ]
    for i in range(1, 13):
        raw_txt = next((t for n, t in numbered if n == i), "")
        txt = _inline_markdown_to_html(raw_txt)
        rows_html.append(
            f"<tr>"
            f"<td style='text-align:center; vertical-align:top;'>{i}</td>"
            f"<td style='text-align:left; vertical-align:top;'>{txt}</td>"
            f"<td style='text-align:left; vertical-align:top;'>______________________</td>"
            f"</tr>"
        )
    
    html = f"""
    <div class="section">Section A [10 marks] (Editing)</div>
    <p style="margin-bottom: 4mm;">
        Carefully read the text below, consisting of 12 lines. The first and last lines are correct. 
        For eight of the lines, there is one grammatical error in each line. There are two more lines with no errors.
    </p>
    <p style="margin-bottom: 6mm;">
        If there is NO error in a line, put a tick (✓) in the Correction column. 
        If the line is incorrect, write the correct word in the Correction column.
    </p>
    <table class="p1a-lines" style="width:100%; border-collapse:collapse; font-size:11pt;">
      {''.join(rows_html)}
    </table>
    """
    return html


def _inline_markdown_to_html(text: str) -> str:
    """Convert minimal markdown (**bold**) to HTML safely, preserving line breaks."""
    # Escape HTML first to avoid injection
    safe = _html.escape(text)
    # Convert **bold** patterns (non-greedy)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    # Convert *italic* patterns while avoiding already-converted **bold**
    safe = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", safe)
    # Preserve line breaks
    safe = safe.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")
    return safe


def render_html_template(
    *,
    paper_format: str,
    section: Optional[str],
    content: str,
    output_html: Path,
    session: Optional[str] = None,
    year: Optional[str] = None,
    duration: Optional[str] = None,
    watermark_text: Optional[str] = None,
    visual_image_path: Optional[Path] = None,
    visual_caption: Optional[str] = None,
) -> Path:
    """Render an HTML template for previewing the paper layout with content injected."""
    env = _env()
    # Prefer format-specific templates; fall back if needed
    template_name = "paper_1.html" if paper_format == "paper_1" else (
        "paper_2.html" if paper_format == "paper_2" else "paper_1.html"
    )
    try:
        template = env.get_template(template_name)
    except TemplateNotFound:
        template = env.get_template("paper_1.html")
    # Build specialized content blocks if applicable
    content_html: Optional[str] = None
    if paper_format == "paper_1" and (section == "section_a"):
        content_html = _build_p1_section_a_html(content)
    # If no specialized block, apply minimal markdown processing for bold asterisks
    if content_html is None:
        # If we have a visual and full paper or target visual section, try to inject image near the right section
        target_header = None
        if visual_image_path:
            if paper_format == "paper_1" and (section in (None, "section_b")):
                # Inject right after "Section B" heading
                target_header = "Section B"
            elif paper_format == "paper_2" and (section in (None, "section_a")):
                target_header = "Section A"
        if visual_image_path and target_header:
            lines = content.splitlines()
            idx = -1
            # Find the Section B/A heading line - be specific to avoid matching wrong sections
            target_lower = target_header.lower()
            logger.info(f"Searching for '{target_header}' in content to place visual | paper_format={paper_format} | section={section}")
            for i, ln in enumerate(lines):
                ln_clean = ln.strip().lower()
                # Match "Section B" or "**Section B" or "Section B [" but not "Section A" or "Section C"
                if target_lower == "section b":
                    # Specifically look for Section B, not A or C
                    if (ln_clean.startswith("section b") or ln_clean.startswith("**section b") or 
                        "section b [" in ln_clean) and "section a" not in ln_clean and "section c" not in ln_clean:
                        idx = i
                        logger.info(f"Found '{target_header}' at line {i}: {ln.strip()[:80]}")
                        break
                elif target_lower == "section a":
                    # Specifically look for Section A, not B or C
                    if (ln_clean.startswith("section a") or ln_clean.startswith("**section a") or 
                        "section a [" in ln_clean) and "section b" not in ln_clean and "section c" not in ln_clean:
                        idx = i
                        logger.info(f"Found '{target_header}' at line {i}: {ln.strip()[:80]}")
                        break
            visual_block = (
                '<div class="visual-block" style="margin:6mm 0; text-align:center;">'
                f'<img src="{str(visual_image_path).replace("\\\\","/")}" alt="Visual stimulus" style="max-width:100%; height:auto;"/>'
                + (f'<div style="font-size:10pt; margin-top:2mm; color:#333;">{_html.escape(visual_caption or "")}</div>' if visual_caption else "")
                + "</div>"
            )
            if idx >= 0:
                # Inject image right after the Section B/A heading
                before = "\n".join(lines[: idx + 1])
                after = "\n".join(lines[idx + 1 :])
                content_html = (
                    f"<div>{_inline_markdown_to_html(before)}</div>{visual_block}<div>{_inline_markdown_to_html(after)}</div>"
                )
                logger.info(f"Visual image placed after '{target_header}' heading at line {idx}")
            else:
                # Fallback: try a broader search for "Section B" anywhere in the line
                idx_fallback = -1
                if target_lower == "section b":
                    for i, ln in enumerate(lines):
                        ln_lower = ln.lower()
                        if "section b" in ln_lower and "section a" not in ln_lower and "section c" not in ln_lower:
                            idx_fallback = i
                            logger.info(f"Found '{target_header}' with broader search at line {i}: {ln.strip()[:80]}")
                            break
                if idx_fallback >= 0:
                    before = "\n".join(lines[: idx_fallback + 1])
                    after = "\n".join(lines[idx_fallback + 1 :])
                    content_html = (
                        f"<div>{_inline_markdown_to_html(before)}</div>{visual_block}<div>{_inline_markdown_to_html(after)}</div>"
                    )
                    logger.info(f"Visual image placed after '{target_header}' heading (fallback) at line {idx_fallback}")
                else:
                    # Last resort: put visual at top of content
                    logger.warning(f"Could not find '{target_header}' heading in content, placing visual at top. Content preview: {content[:200]}")
                    content_html = f"{visual_block}<div>{_inline_markdown_to_html(content)}</div>"
        else:
            content_html = f"<div>{_inline_markdown_to_html(content)}</div>"
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with output_html.open("w", encoding="utf-8") as f:
        f.write(
            template.render(
                paper_format=paper_format,
                section=section,
                content=content,
                content_html=content_html,
                session=session,
                year=year,
                duration=duration,
                watermark_text=watermark_text,
                visual_image_path=str(visual_image_path).replace("\\", "/") if visual_image_path else None,
                visual_caption=visual_caption,
            )
        )
    return output_html


def html_to_pdf(html_path: Path, pdf_path: Path) -> Path:
    """Convert HTML to PDF with best-effort fidelity.
    Try Playwright (Chromium) -> WeasyPrint -> xhtml2pdf."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print("Using Playwright")
        # Highest fidelity: Playwright with headless Chromium.
        # Important: run sync API in a separate thread to avoid FastAPI's running event loop.
        from concurrent.futures import ThreadPoolExecutor
        from playwright.sync_api import sync_playwright  # type: ignore

        def _run_playwright() -> None:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                # Load via file:// URL for maximum compatibility with assets
                file_url = html_path.resolve().as_uri()
                page.goto(file_url, wait_until="load")
                page.emulate_media(media="print")
                page.pdf(
                    path=str(pdf_path),
                    print_background=True,
                    format="A4",
                    margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"},
                    prefer_css_page_size=True,
                )
                browser.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_playwright)
            future.result()
        return pdf_path
    except Exception as e:
        print("Playwright failed", repr(e))
        print("Using WeasyPrint")
        # Next best: WeasyPrint
        try:
            from weasyprint import HTML  # type: ignore

            print("WeasyPrint trying")
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            return pdf_path
        except Exception as e2:
            print("WeasyPrint failed", repr(e2))
            print("Using xhtml2pdf")
            # Fallback: xhtml2pdf
            with html_path.open("r", encoding="utf-8") as source:
                html = source.read()
            with pdf_path.open("wb") as target:
                pisa.CreatePDF(html, dest=target)
            return pdf_path


