#!/usr/bin/env python3
"""Generate all paper types with all difficulty levels for client review."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from app.services.paper_generator import generate_paper, PaperGenerationError


def main():
    """Generate all 9 paper combinations."""
    
    paper_formats = ["paper_1", "paper_2"]
    difficulties = ["foundational", "standard", "advanced"]
    
    # Unique topics for EACH paper + difficulty combination
    topics_map = {
        ("paper_1", "foundational"): ["Technology", "Society"],
        ("paper_1", "standard"): ["Art", "Education"],
        ("paper_1", "advanced"): ["Global Issues", "Environment"],
        ("paper_2", "foundational"): ["AI", "Health"],
        ("paper_2", "standard"): ["Research", "Sports"],
        ("paper_2", "advanced"): ["Cultural Heritage", "Culture"],
    }
    
    results = []
    failed = []
    
    print("=" * 60)
    print("GCE English Paper Generation - Client Review Package")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    total = len(paper_formats) * len(difficulties)
    current = 0
    
    for paper_format in paper_formats:
        for difficulty in difficulties:
            current += 1
            paper_name = paper_format.replace("_", " ").title()
            
            print(f"[{current}/{total}] Generating {paper_name} - {difficulty.title()}...")
            
            try:
                # Get unique topics for this specific paper + difficulty combination
                current_topics = topics_map.get((paper_format, difficulty), ["general knowledge"])
                
                # Only generate answer keys for Paper 1 and Paper 2, not Oral
                should_generate_answer_key = paper_format in ("paper_1", "paper_2")
                
                result = generate_paper(
                    difficulty=difficulty,
                    paper_format=paper_format,
                    section=None,  # Full paper
                    topics=current_topics,
                    additional_instructions=f"Generate a complete, high-quality examination paper suitable for official use. Focus on the themes: {', '.join(current_topics)}.",
                    visual_mode="embed",
                    search_provider="openai",
                    user_id="client_review",  # Special user_id for organization
                    generate_answer_key_flag=should_generate_answer_key,
                )
                
                results.append({
                    "paper_format": paper_format,
                    "difficulty": difficulty,
                    "topics": current_topics,
                    "pdf_path": str(result.pdf_path),
                    "text_path": str(result.text_path),
                    "download_url": result.download_url,
                    "visual_embedded": result.visual_meta is not None,
                    "answer_key_pdf_path": str(result.answer_key_pdf_path) if result.answer_key_pdf_path else None,
                })
                
                print(f"    ✅ Success: {result.pdf_path.name}")
                if result.download_url:
                    print(f"    📥 Download: {result.download_url[:80]}...")
                print()
                
            except PaperGenerationError as e:
                failed.append({
                    "paper_format": paper_format,
                    "difficulty": difficulty,
                    "error": str(e),
                })
                print(f"    ❌ Failed: {e}")
                print()
            except Exception as e:
                failed.append({
                    "paper_format": paper_format,
                    "difficulty": difficulty,
                    "error": str(e),
                })
                print(f"    ❌ Error: {e}")
                print()
    
    # Summary
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print()
    print(f"✅ Successfully generated: {len(results)}/{total} papers")
    if failed:
        print(f"❌ Failed: {len(failed)} papers")
    print()
    
    if results:
        print("Generated Papers:")
        print("-" * 40)
        for r in results:
            paper_name = r["paper_format"].replace("_", " ").title()
            print(f"  • {paper_name} ({r['difficulty']})")
            print(f"    Topics: {', '.join(r['topics'])}")
            print(f"    PDF: {r['pdf_path']}")
            if r['download_url']:
                print(f"    URL: {r['download_url']}")
            print()
    
    if failed:
        print("Failed Papers:")
        print("-" * 40)
        for f in failed:
            paper_name = f["paper_format"].replace("_", " ").title()
            print(f"  • {paper_name} ({f['difficulty']}): {f['error']}")
        print()
    
    # Save summary to file
    summary_path = Path("storage/papers/CLIENT_REVIEW_SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GCE ENGLISH PAPER GENERATION - CLIENT REVIEW PACKAGE\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Papers: {len(results)}/{total}\n\n")
        
        f.write("GENERATED PAPERS:\n")
        f.write("-" * 40 + "\n")
        for r in results:
            paper_name = r["paper_format"].replace("_", " ").title()
            f.write(f"\n{paper_name} - {r['difficulty'].title()}\n")
            f.write(f"  Topics: {', '.join(r['topics'])}\n")
            f.write(f"  Local PDF: {r['pdf_path']}\n")
            if r['download_url']:
                f.write(f"  Download URL: {r['download_url']}\n")
            f.write(f"  Visual Embedded: {'Yes' if r['visual_embedded'] else 'No'}\n")
        
        if failed:
            f.write("\n\nFAILED PAPERS:\n")
            f.write("-" * 40 + "\n")
            for f_item in failed:
                paper_name = f_item["paper_format"].replace("_", " ").title()
                f.write(f"  {paper_name} ({f_item['difficulty']}): {f_item['error']}\n")
    
    print(f"📄 Summary saved to: {summary_path}")
    print()
    
    return len(results), len(failed)


if __name__ == "__main__":
    success, failed = main()
    sys.exit(0 if failed == 0 else 1)

