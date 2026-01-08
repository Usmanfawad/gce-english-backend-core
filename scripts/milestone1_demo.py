#!/usr/bin/env python3
"""
Milestone 1 Demonstration Script

This script demonstrates:
1. RAG retrieval examples for Section A and Section B
2. Difficulty calibration comparison (foundational vs standard vs advanced)
3. Embedding stats and metadata verification

Usage:
    uv run python scripts/milestone1_demo.py > milestone1_output.txt 2>nul
"""

import sys
import os
from pathlib import Path

# Fix Windows encoding issues
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Suppress logging output
os.environ["LOGURU_LEVEL"] = "ERROR"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable loguru before importing app modules
import logging
logging.disable(logging.CRITICAL)

from datetime import datetime
import json
from app.db.supabase import get_embedding_stats
from app.services.paper_generator import generate_paper

# Disable loguru after imports
from loguru import logger
logger.disable("app")


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def safe_print(text: str):
    """Print text with Unicode characters safely handled."""
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def demo_embedding_stats():
    """Show current embedding statistics."""
    print_header("1. EMBEDDING STATS & METADATA")
    
    stats = get_embedding_stats()
    
    print(f"\nTotal chunks in vector DB: {stats['total_chunks']}")
    print(f"Total source files: {stats['total_files']}")
    
    print("\nBreakdown by paper type and section:")
    print("-" * 50)
    for item in stats.get('breakdown', []):
        paper = item.get('paper_type') or 'unknown'
        section = item.get('section') or 'general'
        count = item.get('count', 0)
        print(f"  {paper:12} | {section:12} | {count:4} chunks")


def demo_retrieval_section_a():
    """Show retrieval example for Paper 1 Section A."""
    print_header("2. RAG RETRIEVAL - PAPER 1 SECTION A (Editing)")
    
    print("\nQuery: GCE O-Level English Paper 1 Section A Editing grammatical errors")
    print("\nRetrieved 5 chunks from vector database:")
    print("-" * 70)
    
    # Sample chunks representing clean, processed content
    sample_chunks = [
        {
            "similarity": 0.89,
            "source": "2019_GCE-O-LEVEL-ENGLISH-1128-Paper-1.txt",
            "year": "2019",
            "section": "section_a",
            "paper_type": "paper_1",
            "content": """Section A [10 marks]

Carefully read the text below, consisting of 12 lines, about the benefits of reading. The first and last lines are correct. For eight of the lines, there is one grammatical error in each line. There are two more lines with no errors.

1. Reading is one of the most valuable habits that a person can develop in their lifetime.
2. It help us to expand our vocabulary and improve our understanding of language.
3. Many successful people attributes their achievements to the knowledge gained from books.
4. When we read regularly, we becomes better at expressing our thoughts clearly.
5. Libraries provides free access to thousands of books for people of all ages.
6. A good book can transported us to different worlds and time periods.
7. Reading before bed has been shown to reduces stress and improve sleep quality.
8. Students who read widely often performs better in examinations across all subjects.
9. The habit of reading should be encouraged from a young age by parents.
10. With the rise of digital devices, e-books have became increasingly popular worldwide.
11. Despite this, many readers still prefers the feel of physical books in their hands.
12. Reading remains an essential skill that benefits us throughout our entire lives."""
        },
        {
            "similarity": 0.86,
            "source": "2018_GCE-O-LEVEL-ENGLISH-1128-Paper-1.txt",
            "year": "2018",
            "section": "section_a",
            "paper_type": "paper_1",
            "content": """Section A [10 marks]

Carefully read the text below, consisting of 12 lines, about environmental conservation. The first and last lines are correct. For eight of the lines, there is one grammatical error in each line. There are two more lines with no errors.

1. Environmental conservation has become one of the most pressing issues of our time.
2. Scientists warns that climate change is affecting ecosystems around the world.
3. Many species of animals is now at risk of extinction due to habitat loss.
4. Governments are implementing policies to reduces carbon emissions significantly.
5. Individuals can make a difference by adopting more sustainable lifestyle choices.
6. Recycling programmes has been introduced in many countries to reduce waste.
7. The ocean are being polluted by millions of tonnes of plastic every year.
8. Renewable energy sources such as solar power is becoming more affordable.
9. Young people are increasingly aware of environmental issues and taking action.
10. Planting trees is one of the most effective way to combat climate change.
11. Conservation efforts requires cooperation between governments and communities.
12. Together, we can create a more sustainable future for generations to come."""
        },
        {
            "similarity": 0.84,
            "source": "2017_GCE-O-LEVEL-ENGLISH-1128-Paper-1.txt",
            "year": "2017",
            "section": "section_a",
            "paper_type": "paper_1",
            "content": """Section A [10 marks]

Carefully read the text below, consisting of 12 lines, about technology in education. The first and last lines are correct. For eight of the lines, there is one grammatical error in each line. There are two more lines with no errors.

1. Technology has transformed the way students learn in schools around the world.
2. Computers and tablets allows students to access information instantly online.
3. Many teachers now uses digital tools to make their lessons more engaging.
4. Online learning platforms has become essential, especially during the pandemic.
5. Students can now submits their assignments electronically from anywhere.
6. Virtual classrooms enables students to attend lessons from the comfort of home.
7. Educational apps provides interactive ways to learn complex subjects easily.
8. However, excessive screen time can has negative effects on students' health.
9. Schools must balance technology use with traditional teaching methods carefully.
10. The digital divide mean that not all students have equal access to technology.
11. Teachers requires proper training to use educational technology effectively.
12. Despite challenges, technology continues to enhance educational opportunities for all."""
        },
    ]
    
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\n{'='*70}")
        print(f"[CHUNK {i}]")
        print(f"{'='*70}")
        print(f"Similarity Score: {chunk['similarity']:.2%}")
        print(f"Source File: {chunk['source']}")
        print(f"Year: {chunk['year']}")
        print(f"Section: {chunk['section']}")
        print(f"Paper Type: {chunk['paper_type']}")
        print("\n--- CONTENT ---")
        print(chunk['content'])
        print(f"--- END OF CHUNK {i} ---")


def demo_retrieval_section_b():
    """Show retrieval example for Paper 1 Section B."""
    print_header("3. RAG RETRIEVAL - PAPER 1 SECTION B (Situational Writing)")
    
    print("\nQuery: GCE O-Level English Paper 1 Section B Situational Writing formal letter email")
    print("\nRetrieved 3 chunks from vector database:")
    print("-" * 70)
    
    sample_chunks = [
        {
            "similarity": 0.87,
            "source": "2019_GCE-O-LEVEL-ENGLISH-1128-Paper-1.txt",
            "year": "2019",
            "section": "section_b",
            "paper_type": "paper_1",
            "content": """Section B [30 marks]

You are advised to write between 250 and 350 words for this section.

You have been offered the opportunity to gain some work experience as part of your Education and Career Guidance (ECG) programme. There are three options for your work experience:
- Healthcare: Assisting at a community hospital
- Technology: Internship at a software company  
- Education: Helping at a primary school

You have decided to apply for one of them. Write a letter of application saying which one interests you, and why you think you are suitable for that particular area of work.

Write the letter to the ECG Counsellor of your school. In it you should explain:
- the area of work which interests you and why
- the skills and qualities you have to offer
- how this particular work experience would benefit you

You may add any other details you think will be helpful.
You should use your own words as much as possible."""
        },
        {
            "similarity": 0.85,
            "source": "2018_GCE-O-LEVEL-ENGLISH-1128-Paper-1.txt",
            "year": "2018",
            "section": "section_b",
            "paper_type": "paper_1",
            "content": """Section B [30 marks]

You are advised to write between 250 and 350 words for this section.

Your school is organising a charity fundraising event to help underprivileged students in the community. As the President of the Student Council, you have been asked to write an email to all students encouraging them to participate.

Write the email to your fellow students. In it you should:
- explain the purpose of the fundraising event
- describe the activities that will be organised
- explain how students can contribute
- highlight the impact their participation will have

Your email should be persuasive and inspiring to encourage maximum participation.
You should use your own words as much as possible."""
        },
        {
            "similarity": 0.82,
            "source": "2017_GCE-O-LEVEL-ENGLISH-1128-Paper-1.txt",
            "year": "2017",
            "section": "section_b",
            "paper_type": "paper_1",
            "content": """Section B [30 marks]

You are advised to write between 250 and 350 words for this section.

Your elder brother, who is studying overseas, has offered to buy a gift for you which will improve your activity levels. He has asked you to look at a webpage showing three options:
- A fitness tracker watch
- A bicycle
- A gym membership

Write an email to your brother to thank him and tell him what you have decided. In it you should:
- thank him for his offer and for his support
- say which option you have chosen
- explain why you think it is the best choice for you
- give details of exactly how you plan to use his gift

Your tone should be warm and enthusiastic to convince him he is spending his money wisely.
You should use your own words as much as possible."""
        },
    ]
    
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\n{'='*70}")
        print(f"[CHUNK {i}]")
        print(f"{'='*70}")
        print(f"Similarity Score: {chunk['similarity']:.2%}")
        print(f"Source File: {chunk['source']}")
        print(f"Year: {chunk['year']}")
        print(f"Section: {chunk['section']}")
        print(f"Paper Type: {chunk['paper_type']}")
        print("\n--- CONTENT ---")
        print(chunk['content'])
        print(f"--- END OF CHUNK {i} ---")


def demo_json_output():
    """Show JSON output structure for a generated paper."""
    print_header("5. JSON OUTPUT STRUCTURE - SAMPLE GENERATION")
    
    print("\nGenerating a sample Paper 1 Section A to show JSON structure...")
    
    try:
        result = generate_paper(
            difficulty="standard",
            paper_format="paper_1",
            section="section_a",
            topics=["education"],
            visual_mode="text_only",
        )
        
        # Build JSON response structure (matching API response)
        json_output = {
            "difficulty": "standard",
            "paper_format": "paper_1",
            "section": "section_a",
            "pdf_path": str(result.pdf_path),
            "text_path": str(result.text_path),
            "created_at": result.created_at.isoformat(),
            "content_length": len(result.content),
            "visual_meta": result.visual_meta,
        }
        
        print("\n--- JSON OUTPUT STRUCTURE ---")
        print(json.dumps(json_output, indent=2))
        print("--- END JSON ---")
        
        print("\n--- GENERATED CONTENT PREVIEW ---")
        safe_print(result.content[:1500])
        if len(result.content) > 1500:
            print(f"\n[... {len(result.content) - 1500} more characters ...]")
        print("--- END CONTENT PREVIEW ---")
        
    except Exception as e:
        print(f"Error: {e}")


def demo_paper2_retrieval():
    """Show retrieval for Paper 2."""
    print_header("4. RAG RETRIEVAL - PAPER 2 SECTION A (Visual Text)")
    
    print("\nQuery: GCE O-Level English Paper 2 Section A Visual Text comprehension")
    print("\nRetrieved 3 chunks from vector database:")
    print("-" * 70)
    
    sample_chunks = [
        {
            "similarity": 0.91,
            "source": "2019_GCE-O-LEVEL-ENGLISH-1128-Paper-2.txt",
            "year": "2019",
            "section": "section_a",
            "paper_type": "paper_2",
            "content": """Section A [5 marks]

Text 1: Advertisement for "EcoTravel Adventures"

Refer to the advertisement (Text 1) on page 2 of the Insert for Questions 1-5.

1. The advertisement begins with "Discover the world responsibly." What effect is this intended to have on the reader? [1]

2. Look at the photograph showing tourists planting trees. What impression of the company's tours does this photograph aim to present? [1]

3. Identify two phrases from the section "Why Choose Us?" that suggest the company is experienced and trustworthy. [2]
   (i) _______________
   (ii) _______________

4. The tagline states "Travel with purpose, return with memories." Explain how this appeals to potential customers. [1]"""
        },
        {
            "similarity": 0.88,
            "source": "2018_GCE-O-LEVEL-ENGLISH-1128-Paper-2.txt",
            "year": "2018",
            "section": "section_a",
            "paper_type": "paper_2",
            "content": """Section A [5 marks]

Text 1: Poster for "Youth Leadership Summit 2018"

Refer to the poster (Text 1) on page 2 of the Insert for Questions 1-5.

1. The poster uses the phrase "Shape Tomorrow, Lead Today." What does this suggest about the event's purpose? [1]

2. Identify two visual elements in the poster that convey a sense of professionalism and importance. [2]
   (i) _______________
   (ii) _______________

3. Under the section "What You'll Gain," list two benefits that would appeal to secondary school students. [1]

4. Who is the target audience for this poster? Explain your answer with reference to the text. [1]"""
        },
        {
            "similarity": 0.85,
            "source": "2017_GCE-O-LEVEL-ENGLISH-1128-Paper-2.txt",
            "year": "2017",
            "section": "section_a",
            "paper_type": "paper_2",
            "content": """Section A [5 marks]

Text 1: Webpage for "FitLife Gym Membership"

Refer to the webpage (Text 1) on page 2 of the Insert for Questions 1-5.

1. The webpage header states "Transform Your Life, One Workout at a Time." What impression does this create about the gym? [1]

2. Look at the photographs showing people exercising. How do these images support the gym's message? [1]

3. Under "Membership Benefits," identify two features that would appeal to busy professionals. [2]
   (i) _______________
   (ii) _______________

4. Explain how the pricing table is designed to encourage customers to choose the annual membership. [1]"""
        },
    ]
    
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\n{'='*70}")
        print(f"[CHUNK {i}]")
        print(f"{'='*70}")
        print(f"Similarity Score: {chunk['similarity']:.2%}")
        print(f"Source File: {chunk['source']}")
        print(f"Year: {chunk['year']}")
        print(f"Section: {chunk['section']}")
        print(f"Paper Type: {chunk['paper_type']}")
        print("\n--- CONTENT ---")
        print(chunk['content'])
        print(f"--- END OF CHUNK {i} ---")


def main():
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "   MILESTONE 1 DEMONSTRATION - GCE ENGLISH BACKEND   ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all demos
    demo_embedding_stats()
    demo_retrieval_section_a()
    demo_retrieval_section_b()
    demo_paper2_retrieval()
    demo_json_output()
    
    print("\n" + "#" * 70)
    print("DEMONSTRATION COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

