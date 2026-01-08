#!/usr/bin/env python3
"""
Generate all paper types with all difficulties and answer keys.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from app.services.paper_generator import generate_paper, PaperGenerationError

PAPER_FORMATS = ["paper_1", "paper_2", "oral"]
DIFFICULTIES = ["foundational", "standard", "advanced"]

def generate_all_papers():
    """Generate all paper combinations."""
    print("\n" + "="*60)
    print("GCE ENGLISH - FULL PAPER GENERATION")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    for paper_format in PAPER_FORMATS:
        for difficulty in DIFFICULTIES:
            print(f"\n{'='*60}")
            print(f"Generating {paper_format.upper().replace('_', ' ')} - {difficulty.upper()}")
            print(f"{'='*60}")

            start_time = datetime.now()

            try:
                result = generate_paper(
                    difficulty=difficulty,
                    paper_format=paper_format,
                    section=None,  # Full paper
                    topics=["technology", "environment"],
                    additional_instructions=None,
                    visual_mode="embed",
                    search_provider="openai",
                    generate_answer_key_flag=True,
                )

                duration = (datetime.now() - start_time).total_seconds()

                print(f"\n[SUCCESS]")
                print(f"   PDF: {result.pdf_path}")
                if result.answer_key_pdf_path:
                    print(f"   Answer Key PDF: {result.answer_key_pdf_path}")
                print(f"   Duration: {duration:.1f}s")

                results.append({
                    "status": "success",
                    "paper_format": paper_format,
                    "difficulty": difficulty,
                    "pdf_path": str(result.pdf_path),
                    "answer_key_pdf_path": str(result.answer_key_pdf_path) if result.answer_key_pdf_path else None,
                    "duration_seconds": duration,
                })

            except PaperGenerationError as e:
                duration = (datetime.now() - start_time).total_seconds()
                print(f"\n[FAILED]: {e}")
                results.append({
                    "status": "error",
                    "paper_format": paper_format,
                    "difficulty": difficulty,
                    "error": str(e),
                    "duration_seconds": duration,
                })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    success_count = sum(1 for r in results if r["status"] == "success")
    total_duration = sum(r.get("duration_seconds", 0) for r in results)

    print(f"Total papers generated: {success_count}/{len(results)}")
    print(f"Total time: {total_duration:.1f}s")

    print("\n=== GENERATED FILES ===")
    for r in results:
        if r["status"] == "success":
            print(f"\n[OK] {r['paper_format'].upper().replace('_', ' ')} - {r['difficulty'].upper()}")
            print(f"   PDF: {r['pdf_path']}")
            if r.get("answer_key_pdf_path"):
                print(f"   Answer Key: {r['answer_key_pdf_path']}")
        else:
            print(f"\n[FAIL] {r['paper_format'].upper().replace('_', ' ')} - {r['difficulty'].upper()}")
            print(f"   Error: {r.get('error', 'Unknown')}")

    print("\n" + "="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    return results

if __name__ == "__main__":
    results = generate_all_papers()
