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
    # NO visible borders - use invisible table for layout only
    rows_html = [
        "<tr>"
        "<th style='text-align:center; width:12mm; font-weight:bold; border:none; padding:4px 8px;'>Line</th>"
        "<th style='text-align:left; font-weight:bold; border:none; padding:4px 8px;'>Text</th>"
        "<th style='text-align:center; width:50mm; font-weight:bold; border:none; padding:4px 8px;'>Correction</th>"
        "</tr>"
    ]
    for i in range(1, 13):
        raw_txt = next((t for n, t in numbered if n == i), "")
        txt = _inline_markdown_to_html(raw_txt)
        rows_html.append(
            f"<tr>"
            f"<td style='text-align:center; vertical-align:top; border:none; padding:4px 8px;'>{i}</td>"
            f"<td style='text-align:left; vertical-align:top; border:none; padding:4px 8px;'>{txt}</td>"
            f"<td style='text-align:left; vertical-align:top; border:none; padding:4px 8px;'>______________________</td>"
            f"</tr>"
        )
    
    html = f"""
    <div style="page-break-before:always;"></div>
    <div class="section" style="font-weight:bold; font-size:14pt; margin:8mm 0 6mm 0; text-transform:uppercase; border-bottom:1px solid #000; padding-bottom:2mm;">SECTION A <span style="font-weight:normal; font-size:12pt;">[10 marks]</span></div>
    <p style="margin-bottom: 4mm;">
        The following passage contains some grammatical errors. Each of the 12 lines may or may not contain one error.
    </p>
    <p style="margin-bottom: 4mm;">
        If there is an error in a line, write the correct word in the <b>Correction</b> column.
        If the line is correct, put a tick (✓) in the <b>Correction</b> column.
    </p>
    <p style="margin-bottom: 6mm; font-style: italic;">
        The first and last lines are correct. There are <b>eight</b> errors in total.
    </p>
    <table class="p1a-lines" style="width:100%; border-collapse:collapse; border:none; font-size:11pt;">
      {''.join(rows_html)}
    </table>
    """
    return html


def _inline_markdown_to_html(text: str, escape_html: bool = True) -> str:
    """Convert minimal markdown (**bold**) to HTML safely, preserving line breaks."""
    if escape_html:
        safe = _html.escape(text)
    else:
        safe = text
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", safe)
    parts = re.split(r'(<[^>]+>)', safe)
    processed_parts = []
    for part in parts:
        if part.startswith('<') and part.endswith('>'):
            processed_parts.append(part)
        else:
            processed_parts.append(part.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>"))
    return ''.join(processed_parts)


def _enhance_section_headers(content: str) -> str:
    """Enhance section headers with proper marks display and page breaks.

    Works on content that has already been processed by _inline_markdown_to_html,
    so **bold** has become <b>bold</b>.

    Uses inline styles for maximum compatibility with PDF converters.
    """
    import re

    # Inline styles for section headers (more compatible than CSS classes)
    section_header_style = 'style="font-weight:bold; font-size:14pt; margin:8mm 0 6mm 0; text-transform:uppercase; border-bottom:1px solid #000; padding-bottom:2mm; display:block; page-break-before:always;"'
    marks_style = 'style="font-weight:normal; font-size:12pt;"'
    component_header_style = 'style="font-weight:bold; font-size:14pt; margin:10mm 0 6mm 0; text-transform:uppercase; border-bottom:2px solid #000; padding-bottom:3mm; display:block;"'
    component_header_break_style = 'style="font-weight:bold; font-size:14pt; margin:10mm 0 6mm 0; text-transform:uppercase; border-bottom:2px solid #000; padding-bottom:3mm; display:block; page-break-before:always;"'

    # Pattern to match section headers - both HTML bold format and plain format
    section_patterns = [
        # HTML bold format (from markdown conversion): <b>Section A [10 marks]</b>
        (r'<b>Section\s+([ABC])\s*\[(\d+)\s*marks?\]</b>', f'<div {section_header_style}>SECTION \\1 <span {marks_style}>[\\2 marks]</span></div>'),
        (r'<b>Section\s+([ABC])</b>', f'<div {section_header_style}>SECTION \\1</div>'),
        # Plain format (no bold)
        (r'(?<![<\w])Section\s+([ABC])\s*\[(\d+)\s*marks?\](?![^<]*>)', f'<div {section_header_style}>SECTION \\1 <span {marks_style}>[\\2 marks]</span></div>'),

        # Oral exam components - HTML bold format
        (r'<b>READING\s+ALOUD\s*\[(\d+)\s*marks?\]</b>', f'<div {component_header_style}>READING ALOUD <span {marks_style}>[\\1 marks]</span></div>'),
        (r'<b>Reading\s+Aloud\s*\[(\d+)\s*marks?\]</b>', f'<div {component_header_style}>READING ALOUD <span {marks_style}>[\\1 marks]</span></div>'),
        (r'READING\s+ALOUD\s*\[(\d+)\s*marks?\]', f'<div {component_header_style}>READING ALOUD <span {marks_style}>[\\1 marks]</span></div>'),

        (r'<b>STIMULUS-BASED\s+CONVERSATION\s*\[(\d+)\s*marks?\]</b>', f'<div {component_header_break_style}>STIMULUS-BASED CONVERSATION <span {marks_style}>[\\1 marks]</span></div>'),
        (r'<b>Stimulus-Based\s+Conversation\s*\[(\d+)\s*marks?\]</b>', f'<div {component_header_break_style}>STIMULUS-BASED CONVERSATION <span {marks_style}>[\\1 marks]</span></div>'),
        (r'STIMULUS-BASED\s+CONVERSATION\s*\[(\d+)\s*marks?\]', f'<div {component_header_break_style}>STIMULUS-BASED CONVERSATION <span {marks_style}>[\\1 marks]</span></div>'),

        (r'<b>GENERAL\s+CONVERSATION\s*\[(\d+)\s*marks?\]</b>', f'<div {component_header_break_style}>GENERAL CONVERSATION <span {marks_style}>[\\1 marks]</span></div>'),
        (r'<b>General\s+Conversation\s*\[(\d+)\s*marks?\]</b>', f'<div {component_header_break_style}>GENERAL CONVERSATION <span {marks_style}>[\\1 marks]</span></div>'),
        (r'GENERAL\s+CONVERSATION\s*\[(\d+)\s*marks?\]', f'<div {component_header_break_style}>GENERAL CONVERSATION <span {marks_style}>[\\1 marks]</span></div>'),

        # Also handle "Part 1:", "Part 2:", "Part 3:" for oral components
        (r'<b>Part\s+(\d+):\s*Reading\s+Aloud</b>', f'<div {component_header_style}>PART \\1: READING ALOUD</div>'),
        (r'<b>Part\s+(\d+):\s*Stimulus-Based\s+Conversation</b>', f'<div {component_header_break_style}>PART \\1: STIMULUS-BASED CONVERSATION</div>'),
        (r'<b>Part\s+(\d+):\s*General\s+Conversation</b>', f'<div {component_header_break_style}>PART \\1: GENERAL CONVERSATION</div>'),
    ]

    enhanced = content
    for pattern, replacement in section_patterns:
        enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)

    return enhanced


def _add_section_styles() -> str:
    """Return additional CSS for section formatting."""
    return """
    <style>
    .section-header {
        font-weight: bold;
        font-size: 14pt;
        margin: 8mm 0 6mm 0;
        text-transform: uppercase;
        border-bottom: 1px solid #000;
        padding-bottom: 2mm;
        clear: both;
        display: block;
    }
    .section-header .marks {
        font-weight: normal;
        font-size: 12pt;
    }
    .component-header {
        font-weight: bold;
        font-size: 14pt;
        margin: 10mm 0 6mm 0;
        text-transform: uppercase;
        border-bottom: 2px solid #000;
        padding-bottom: 3mm;
        clear: both;
        display: block;
    }
    .component-marks {
        font-weight: normal;
        font-size: 12pt;
    }
    .page-break-before {
        page-break-before: always;
        margin-top: 0;
    }
    .question-mark {
        font-weight: bold;
        color: #333;
    }
    /* Prevent text overlap */
    p, div {
        position: relative;
        z-index: 1;
    }
    /* Ensure proper line spacing */
    .content {
        line-height: 1.5;
    }
    .content br {
        display: block;
        margin: 2mm 0;
        content: "";
    }
    /* Visual block spacing */
    .visual-block {
        margin: 8mm 0;
        page-break-inside: avoid;
    }
    </style>
    """


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
    # Prefer format-specific templates
    if paper_format == "paper_1":
        template_name = "paper_1.html"
    elif paper_format == "paper_2":
        template_name = "paper_2.html"
    elif paper_format == "oral":
        template_name = "oral.html"
    else:
        template_name = "paper_1.html"
    try:
        template = env.get_template(template_name)
    except TemplateNotFound:
        template = env.get_template("paper_1.html")
    # Build specialized content blocks if applicable
    content_html: Optional[str] = None

    # For Paper 1 Section A only (single section), render the editing table with correction blanks
    if paper_format == "paper_1" and section == "section_a":
        section_a_html = _build_p1_section_a_html(content)
        if section_a_html:
            content_html = f"{_add_section_styles()}{section_a_html}"
    
    # For full Paper 1, render Section A as table and Sections B/C normally
    if paper_format == "paper_1" and section is None and content_html is None:
        section_b_match = re.search(r'(?:\*\*)?Section\s+B', content, re.IGNORECASE)
        if section_b_match:
            section_a_content = content[:section_b_match.start()]
            rest_content = content[section_b_match.start():]
            
            section_a_rendered = _build_p1_section_a_html(section_a_content)
            
            if section_a_rendered:
                # Build visual block for Section B if visual provided
                visual_block_html = ""
                if visual_image_path:
                    visual_block_html = (
                        '<div class="visual-block" style="margin:8mm 0; text-align:center; page-break-inside:avoid;">'
                        f'<img src="{str(visual_image_path).replace(chr(92), "/")}" alt="Visual stimulus" style="max-width:100%; max-height:600px; height:auto; border:1px solid #ccc; display:block; margin:0 auto;"/>'
                        + (f'<div style="font-size:10pt; margin-top:3mm; color:#333; font-style:italic;">{_html.escape(visual_caption or "")}</div>' if visual_caption else "")
                        + "</div>"
                    )
                
                # Find Section B header line to inject visual after it
                rest_lines = rest_content.split('\n')
                section_b_header_idx = -1
                for i, line in enumerate(rest_lines):
                    if re.search(r'(?:\*\*)?Section\s+B', line, re.IGNORECASE):
                        section_b_header_idx = i
                        break
                
                if section_b_header_idx >= 0 and visual_block_html:
                    # Insert visual after Section B header
                    before_visual = '\n'.join(rest_lines[:section_b_header_idx + 1])
                    after_visual = '\n'.join(rest_lines[section_b_header_idx + 1:])
                    
                    before_processed = _inline_markdown_to_html(before_visual, escape_html=True)
                    before_enhanced = _enhance_section_headers(before_processed)
                    after_processed = _inline_markdown_to_html(after_visual, escape_html=True)
                    after_enhanced = _enhance_section_headers(after_processed)
                    
                    rest_html = f"<div>{before_enhanced}</div>{visual_block_html}<div>{after_enhanced}</div>"
                else:
                    rest_processed = _inline_markdown_to_html(rest_content, escape_html=True)
                    rest_enhanced = _enhance_section_headers(rest_processed)
                    rest_html = f"<div>{rest_enhanced}</div>"
                
                content_html = f"{_add_section_styles()}{section_a_rendered}{rest_html}"
    
    # If no specialized block, apply processing
    if content_html is None:
        # Visual block to inject (if applicable)
        visual_block = ""
        target_header = None
        
        if visual_image_path:
            if paper_format == "paper_1" and (section in (None, "section_b")):
                target_header = "Section B"
            elif paper_format == "paper_2" and (section in (None, "section_a")):
                target_header = "Section A"
            elif paper_format == "oral":
                target_header = "Stimulus-Based Conversation"
            
            visual_block = (
                '<div class="visual-block" style="margin:8mm 0; text-align:center; page-break-inside:avoid;">'
                f'<img src="{str(visual_image_path).replace(chr(92), "/")}" alt="Visual stimulus" style="max-width:100%; max-height:600px; height:auto; border:1px solid #ccc; display:block; margin:0 auto;"/>'
                + (f'<div style="font-size:10pt; margin-top:3mm; color:#333; font-style:italic;">{_html.escape(visual_caption or "")}</div>' if visual_caption else "")
                + "</div>"
            )
        
        # Step 1: Find section header and inject visual AFTER the header line
        # Work with raw content that still has newlines
        if visual_image_path and target_header:
            # Use regex to find section header position in raw content
            # Match the FULL header line including marks, e.g., "**Section B [30 marks]**"
            if target_header.lower() == "section b":
                # Find Section B header line - look for the complete header pattern
                section_pattern = re.compile(r'^[\s\*]*(Section\s+B[^\n]*)', re.IGNORECASE | re.MULTILINE)
            elif target_header.lower() == "stimulus-based conversation":
                # Find Stimulus-Based Conversation header for oral exams
                section_pattern = re.compile(r'^[\s\*]*(Stimulus[- ]Based\s+Conversation[^\n]*|Part\s+2[:\s][^\n]*)', re.IGNORECASE | re.MULTILINE)
            else:  # Section A for Paper 2
                section_pattern = re.compile(r'^[\s\*]*(Section\s+A[^\n]*)', re.IGNORECASE | re.MULTILINE)
            
            match = section_pattern.search(content)
            
            if match:
                # Get the full header line
                header_line = match.group(0).strip()
                match_end = match.end()
                
                # Find the newline that ends this header line
                newline_pos = content.find('\n', match_end)
                if newline_pos == -1:
                    newline_pos = len(content)
                
                # IMPORTANT: Split so that:
                # - before_content = everything INCLUDING the header line (up to and including the header)
                # - after_content = everything AFTER the header line (starting from the newline)
                # This puts the visual image AFTER the Section B header
                split_point = newline_pos
                
                before_content = content[:split_point]
                after_content = content[split_point:]
                
                logger.info(f"Found '{target_header}' | header: '{header_line[:60]}' | split_point={split_point} | before ends with: '{before_content[-50:]}'")
                
                # Process each part separately
                before_processed = _inline_markdown_to_html(before_content, escape_html=True)
                before_enhanced = _enhance_section_headers(before_processed)
                
                after_processed = _inline_markdown_to_html(after_content, escape_html=True)
                after_enhanced = _enhance_section_headers(after_processed)
                
                content_html = (
                    f"{_add_section_styles()}"
                    f"<div>{before_enhanced}</div>"
                    f"{visual_block}"
                    f"<div>{after_enhanced}</div>"
                )
                logger.info(f"Visual image placed AFTER '{target_header}' heading")
            else:
                # Fallback: put visual right before Section B/A content if we can find it differently
                logger.warning(f"Could not find '{target_header}' with regex. Trying line-by-line search...")
                
                # Try line-by-line as backup
                lines = content.splitlines()
                idx = -1
                for i, ln in enumerate(lines):
                    ln_lower = ln.lower()
                    if target_header.lower() == "section b":
                        if "section b" in ln_lower and "section a" not in ln_lower:
                            idx = i
                            break
                    elif target_header.lower() == "stimulus-based conversation":
                        if "stimulus" in ln_lower and "conversation" in ln_lower:
                            idx = i
                            break
                    elif target_header.lower() == "section a":
                        if "section a" in ln_lower and "section b" not in ln_lower:
                            idx = i
                            break
                
                if idx >= 0:
                    before_lines = lines[:idx + 1]
                    after_lines = lines[idx + 1:]
                    
                    before_processed = _inline_markdown_to_html("\n".join(before_lines), escape_html=True)
                    before_enhanced = _enhance_section_headers(before_processed)
                    
                    after_processed = _inline_markdown_to_html("\n".join(after_lines), escape_html=True)
                    after_enhanced = _enhance_section_headers(after_processed)
                    
                    content_html = (
                        f"{_add_section_styles()}"
                        f"<div>{before_enhanced}</div>"
                        f"{visual_block}"
                        f"<div>{after_enhanced}</div>"
                    )
                    logger.info(f"Visual image placed after '{target_header}' heading (line search)")
                else:
                    # Last resort: put visual at top
                    logger.warning(f"Could not find '{target_header}' heading at all. Content preview: {content[:300]}")
                    processed_content = _inline_markdown_to_html(content, escape_html=True)
                    enhanced_content = _enhance_section_headers(processed_content)
                    content_html = f"{_add_section_styles()}{visual_block}<div>{enhanced_content}</div>"
        else:
            # No visual to inject - just process normally
            processed_content = _inline_markdown_to_html(content, escape_html=True)
            enhanced_content = _enhance_section_headers(processed_content)
            content_html = f"{_add_section_styles()}<div>{enhanced_content}</div>"
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


