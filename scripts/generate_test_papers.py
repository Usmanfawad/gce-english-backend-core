#!/usr/bin/env python3
"""
Test script to generate sample GCE English papers.

Generates:
- 1 complete Paper 1 (Writing)
- 1 complete Paper 2 (Comprehension)

Topics are randomly selected from a predefined list.

Usage:
    python scripts/generate_test_papers.py
    
    # Or with uv
    uv run python scripts/generate_test_papers.py
"""

import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from app.services.paper_generator import generate_paper, PaperGenerationError


# ============================================
# TOPIC LIST - Add/remove topics as needed
# ============================================
TOPICS = [
    # Technology
    "artificial intelligence",
    "social media",
    "smartphones",
    "online learning",
    "cybersecurity",
    "virtual reality",
    
    # Environment
    "climate change",
    "renewable energy",
    "wildlife conservation",
    "plastic pollution",
    "sustainable living",
    "deforestation",
    
    # Society
    "mental health",
    "work-life balance",
    "cultural diversity",
    "volunteerism",
    "urban development",
    "public transportation",
    
    # Education
    "lifelong learning",
    "exam stress",
    "extracurricular activities",
    "reading habits",
    "creativity in education",
    
    # Lifestyle
    "healthy eating",
    "fitness and exercise",
    "travel and tourism",
    "hobbies and interests",
    "time management",
    
    # Youth Issues
    "peer pressure",
    "career choices",
    "family relationships",
    "friendship",
    "online safety",
]

# Difficulty levels to cycle through
DIFFICULTIES = ["foundational", "standard", "advanced"]


def get_random_topics(count: int = 2) -> list[str]:
    """Select random topics from the list."""
    return random.sample(TOPICS, min(count, len(TOPICS)))


def generate_test_paper(
    paper_format: str,
    difficulty: str,
    topics: list[str],
) -> dict:
    """Generate a single test paper and return results."""
    print(f"\n{'='*60}")
    print(f"Generating {paper_format.upper().replace('_', ' ')}")
    print(f"Difficulty: {difficulty}")
    print(f"Topics: {', '.join(topics)}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        result = generate_paper(
            difficulty=difficulty,
            paper_format=paper_format,
            section=None,  # Generate full paper
            topics=topics,
            additional_instructions=None,
            visual_mode="embed",
            search_provider="openai",
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ SUCCESS!")
        print(f"   PDF: {result.pdf_path}")
        print(f"   TXT: {result.text_path}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Content length: {len(result.content)} chars")
        
        if result.visual_meta:
            print(f"   Visual: {result.visual_meta.get('title', 'N/A')}")
        
        return {
            "status": "success",
            "paper_format": paper_format,
            "difficulty": difficulty,
            "topics": topics,
            "pdf_path": str(result.pdf_path),
            "text_path": str(result.text_path),
            "duration_seconds": duration,
            "content_length": len(result.content),
        }
        
    except PaperGenerationError as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n❌ FAILED: {e}")
        return {
            "status": "error",
            "paper_format": paper_format,
            "difficulty": difficulty,
            "topics": topics,
            "error": str(e),
            "duration_seconds": duration,
        }


def main():
    """Main test function."""
    print("\n" + "="*60)
    print("GCE ENGLISH PAPER GENERATION TEST")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Generate Paper 1
    paper1_difficulty = random.choice(DIFFICULTIES)
    paper1_topics = get_random_topics(2)
    result1 = generate_test_paper("paper_1", paper1_difficulty, paper1_topics)
    results.append(result1)
    
    # Generate Paper 2
    paper2_difficulty = random.choice(DIFFICULTIES)
    paper2_topics = get_random_topics(2)
    result2 = generate_test_paper("paper_2", paper2_difficulty, paper2_topics)
    results.append(result2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    total_duration = sum(r.get("duration_seconds", 0) for r in results)
    
    print(f"Total papers generated: {success_count}/{len(results)}")
    print(f"Total time: {total_duration:.1f}s")
    
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        print(f"\n{status_icon} {r['paper_format'].upper().replace('_', ' ')}")
        print(f"   Difficulty: {r['difficulty']}")
        print(f"   Topics: {', '.join(r['topics'])}")
        if r["status"] == "success":
            print(f"   PDF: {r['pdf_path']}")
        else:
            print(f"   Error: {r.get('error', 'Unknown')}")
    
    print("\n" + "="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Return exit code based on success
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit(main())

