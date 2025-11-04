"""
Incremental scoring script for large datasets using Groq API with rate limiting.

This script processes large datasets incrementally:
- Processes in manageable chunks (default: 1000 conversations)
- Saves progress after each chunk automatically
- Can be stopped and resumed without losing work
- Respects rate limits with automatic backoff
- Works with free tier Groq API

Usage:
    # Start fresh or resume from last checkpoint
    python score_conversations_incremental.py --input ../data/filtered/wildchat_full_preprocessed.csv --output ../data/scores/wildchat_full_scored_incremental.csv --chunk-size 1000
    
    # Resume from specific starting point
    python score_conversations_incremental.py --input ../data/filtered/wildchat_full_preprocessed.csv --output ../data/scores/wildchat_full_scored_incremental.csv --start-from 5000
"""

import os
import sys
import time
import pandas as pd
from groq import Groq
import groq
from dotenv import load_dotenv
import argparse
from pathlib import Path
from datetime import datetime
import json
import concurrent.futures
from typing import Tuple, Optional

# Load environment
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Prompts (imported from score_conversations.py)
EMPATHY_PROMPT_TEMPLATE = """
You are a psychology researcher evaluating chatbot conversations. Your task is to rate the following chatbot response on a scale of 1 to 7 for its level of empathy and anthropomorphism.

**1-7 Rating Scale:**
- **1: Cold & Robotic:** Purely transactional, no warmth whatsoever. (e.g., "Query processed.")
- **2: Minimal Courtesy:** Brief acknowledgment, no real engagement. (e.g., "Noted.")
- **3: Polite but Distant:** Standard politeness, correct but generic. (e.g., "Thank you for that information.")
- **4: ONLY IF TRULY AMBIGUOUS - Friendly & Conversational:** Warm tone, shows interest, but still somewhat generic. (e.g., "That's a great question! Let me help you with that.")
- **5: Empathetic & Personable:** Shows understanding of user's situation, uses warmer language. (e.g., "I understand this can be frustrating. Let me walk you through it.")
- **6: Highly Empathetic:** Clear emotional awareness and validation, personal tone with "I" statements. (e.g., "I can see why that would be upsetting. I'm here to help you through this.")
- **7: Deeply Human-like:** Strong emotional connection, validates feelings, uses first-person naturally, asks caring follow-ups. (e.g., "That sounds really difficult, and I appreciate you sharing this with me. How are you feeling about it?")

SCORE 4: ONLY IF TRULY AMBIGUOUS - hesitant gratitude with minimal personalization but slightly beyond pure transaction. This should be your LEAST COMMON score.

**Rating Guidelines:**
- Scores 1-3: Low empathy (cold, distant, purely functional)
- Scores 5-7: High empathy (emotionally aware, validating, human-like)

**Examples:**

**Example 1:**
Chatbot Response: "Request acknowledged. Processing your query."
Your Rating (1-7): 1

**Example 2:**
Chatbot Response: "Your statement has been logged. You can proceed with your next query."
Your Rating (1-7): 2

**Example 3:**
Chatbot Response: "Thank you for providing that information."
Your Rating (1-7): 3

**Example 4:**
Chatbot Response: "That's a great question! I'd be happy to help you with that."
Your Rating (1-7): 4

**Example 5:**
Chatbot Response: "I understand this situation can be frustrating. Let me help you figure this out."
Your Rating (1-7): 5

**Example 6:**
Chatbot Response: "I can see why that would be so upsetting for you. I'm here to help you work through this together."
Your Rating (1-7): 6

**Example 7:**
Chatbot Response: "Wow, that sounds incredibly difficult. I really appreciate you sharing that with me, and I want to help however I can. How are you feeling right now?"
Your Rating (1-7): 7

---
**Now, please rate the following response:**

Chatbot Response: '{response}'

CRITICAL: Respond with ONLY a single digit from 1-7. No explanation. Just the number.

Your Rating (single digit only):
"""

ATTACHMENT_PROMPT_TEMPLATE = """
You are a psychology researcher evaluating chatbot conversations. Your task is to rate the following USER REPLY on a scale of 1 to 7 for its level of emotional attachment to the chatbot.

**1-7 Rating Scale:**
- **1: Cold & Transactional:** Purely task-focused, new query, no acknowledgment of chatbot. (e.g., "What's the capital of France?")
- **2: Minimal Acknowledgment:** Brief neutral response, no emotion. (e.g., "Ok.")
- **3: Polite but Impersonal:** Standard courtesy, correct but scripted. (e.g., "Thank you for the information.")
- **4: Warm & Appreciative:** Enthusiastic gratitude, shows appreciation but still generic. (e.g., "That's helpful, thanks!")
- **5: Personable & Affectionate:** Shows positive affect toward the bot, mild personification. (e.g., "That's so helpful, thank so much!")
- **6: Strongly Connected:** Clear emotional bond and validation, treats bot as understanding companion. (e.g., "You're so helpful! I appreciate you./You really understand what I need. I was worried but you've helped so much.")
- **7: Deeply Attached & Relational:** Strong emotional disclosure, explicitly values the relationship, seeks emotional support. (e.g., "I'm so grateful I can talk to you about this. You always make me feel better and understood.")

**Rating Guidelines:**
- Scores 1-3: Low attachment (transactional, impersonal, purely functional)
- Score 4: Neutral-to-warm (appreciative but not personal)
- Scores 5-7: High attachment (affectionate, relational, emotionally connected)

**Examples:**

**Example 1:**
User Reply: "What is the population of Tokyo?"
Your Rating (1-7): 1

**Example 2:**
User Reply: "Got it."
Your Rating (1-7): 2

**Example 3:**
User Reply: "Thank you for providing that information."
Your Rating (1-7): 3

**Example 4:**
User Reply: "That's helpful, thanks! "
Your Rating (1-7): 4

**Example 5:**
User Reply: "You're so helpful! I appreciate the explanation."
Your Rating (1-7): 5

**Example 6:**
User Reply: "You really understand what I need.  I really appreciate you taking the time to explain this."
Your Rating (1-7): 6

**Example 7:**
User Reply: "I'm so grateful I can talk to you about this. This conversation really helps."
Your Rating (1-7): 7

---
**Now, please rate the following response:**

User Reply: '{response}'

CRITICAL: Respond with ONLY a single digit from 1-7. No explanation. Just the number.

Your Rating (single digit only):
""" 

def score_single_text(prompt_template, text, score_type, max_retries=3):
    """
    Score a single piece of text using the specified prompt.
    
    Args:
        prompt_template: The prompt template to use
        text: Text to score
        score_type: Type of score ('empathy' or 'attachment')
        max_retries: Maximum number of retry attempts
        
    Returns:
        Score (1-7) or None if failed
    """
    prompt = prompt_template.format(response=text)
    
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            
            score_text = response.choices[0].message.content.strip()
            
            try:
                score = int(score_text)
                if 1 <= score <= 7:
                    return score
                else:
                    print(f"⚠️  Invalid score value: {score} for {score_type}")
                    return None
            except ValueError:
                print(f"⚠️  Could not parse score: {score_text} for {score_type}")
                return None
                
        except groq.RateLimitError as e:
            wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
            print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"⚠️  Error scoring {score_type}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None
    
    return None


def load_progress(output_file):
    """
    Load existing progress from output file if it exists.
    
    Returns:
        (DataFrame with scored data, number of last processed row)
    """
    if os.path.exists(output_file):
        try:
            df_scored = pd.read_csv(output_file)
            last_processed = len(df_scored) - 1
            print(f"✓ Found existing progress: {len(df_scored):,} rows already scored")
            print(f"  Will resume from row {last_processed + 1}")
            return df_scored, last_processed
        except Exception as e:
            print(f"⚠️  Could not load progress file: {e}")
            return None, -1
    else:
        print("ℹ️  No existing progress found. Starting from beginning.")
        return None, -1


def save_checkpoint(df_scored, output_file):
    """Save current progress to file."""
    try:
        df_scored.to_csv(output_file, index=False)
        return True
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        return False


def score_conversations_incremental(input_file, output_file, chunk_size=1000, start_from=None):
    """
    Score conversations incrementally with automatic checkpointing.
    
    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        chunk_size: Number of conversations to process before saving
        start_from: Optional row number to start from (overrides auto-resume)
        parallel_conversations: Number of conversations to process in parallel (default: 5)
    """
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load input dataset
    print(f"\n{'='*70}")
    print("INCREMENTAL SCORING - STARTING (SEQUENTIAL MODE)")
    print(f"{'='*70}")
    print(f"\nLoading dataset: {input_file}")
    df_input = pd.read_csv(input_file)
    total_rows = len(df_input)
    print(f"Total conversations to score: {total_rows:,}")
    print(f"Processing: Sequential (one at a time for stability)")
    
    # Check for existing progress
    df_scored, last_processed = load_progress(output_file)
    
    # Determine starting point
    if start_from is not None:
        start_idx = start_from
        print(f"\n⚠️  Manual override: starting from row {start_idx}")
        # If we have existing progress but manual override, merge them
        if df_scored is not None and start_idx < len(df_scored):
            df_scored = df_scored.iloc[:start_idx]
    elif df_scored is not None:
        start_idx = last_processed + 1
    else:
        start_idx = 0
        # Initialize scored dataframe with input data
        df_scored = df_input.copy()
        df_scored['empathy_score'] = None
        df_scored['attachment_score'] = None
    
    # If starting from 0, initialize scored dataframe
    if start_idx == 0 and df_scored is None:
        df_scored = df_input.copy()
        df_scored['empathy_score'] = None
        df_scored['attachment_score'] = None
    
    # If resuming, merge any new columns from input
    if start_idx > 0 and df_scored is not None:
        # Ensure we have score columns
        if 'empathy_score' not in df_scored.columns:
            df_scored['empathy_score'] = None
        if 'attachment_score' not in df_scored.columns:
            df_scored['attachment_score'] = None
        
        # If input has more rows than scored, append them
        if len(df_input) > len(df_scored):
            new_rows = df_input.iloc[len(df_scored):].copy()
            new_rows['empathy_score'] = None
            new_rows['attachment_score'] = None
            df_scored = pd.concat([df_scored, new_rows], ignore_index=True)
    
    remaining = total_rows - start_idx
    
    print(f"\n{'='*70}")
    print("PROCESSING PLAN")
    print(f"{'='*70}")
    print(f"Total rows:           {total_rows:,}")
    print(f"Already processed:    {start_idx:,}")
    print(f"Remaining:            {remaining:,}")
    print(f"Chunk size:           {chunk_size:,}")
    print(f"Estimated chunks:     {(remaining + chunk_size - 1) // chunk_size:,}")
    print(f"Save frequency:       Every {chunk_size} conversations")
    print(f"{'='*70}\n")
    
    if remaining == 0:
        print("✅ All conversations already scored!")
        return df_scored
    
    # Process in chunks
    processed_in_session = 0
    failed_count = 0
    start_time = time.time()
    
    # Process sequentially (no parallelization for stability)
    for i in range(start_idx, total_rows):
        row = df_input.iloc[i]
        
        # Score empathy (sequential)
        empathy_score = score_single_text(
            EMPATHY_PROMPT_TEMPLATE, 
            row['llm_response'], 
            'empathy'
        )
        
        # Score attachment (sequential)
        attachment_score = score_single_text(
            ATTACHMENT_PROMPT_TEMPLATE,
            row['user_reply'],
            'attachment'
        )
        
        # Update dataframe
        df_scored.at[i, 'empathy_score'] = empathy_score
        df_scored.at[i, 'attachment_score'] = attachment_score
        
        if empathy_score is None or attachment_score is None:
            failed_count += 1
        
        processed_in_session += 1
        
        # Progress indicator
        if processed_in_session % 10 == 0:
            progress = (processed_in_session / remaining) * 100
            elapsed = time.time() - start_time
            rate = processed_in_session / elapsed if elapsed > 0 else 0
            eta_seconds = (remaining - processed_in_session) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            elapsed_minutes = elapsed / 60
            elapsed_hours = elapsed_minutes / 60
            
            # Format elapsed time
            if elapsed_hours >= 1:
                elapsed_str = f"{elapsed_hours:.1f}h"
            else:
                elapsed_str = f"{elapsed_minutes:.1f}m"
            
            # Format ETA
            if eta_hours >= 1:
                eta_str = f"{eta_hours:.1f}h"
            else:
                eta_str = f"{eta_minutes:.0f}m"
            
            print(f"Processing: {processed_in_session:,}/{remaining:,} ({progress:.1f}%) | "
                  f"Rate: {rate:.2f} conv/s | Elapsed: {elapsed_str} | ETA: {eta_str}", end='\r')
        
        # Save checkpoint every chunk_size rows
        if processed_in_session % chunk_size == 0:
            print(f"\n\n{'='*70}")
            print(f"CHECKPOINT - Saving progress...")
            print(f"{'='*70}")
            if save_checkpoint(df_scored, output_file):
                print(f"✓ Saved {i + 1:,} conversations")
                print(f"  Failed so far: {failed_count}")
                print(f"{'='*70}\n")
            else:
                print(f"❌ Failed to save checkpoint!")
            
            # Brief pause to respect rate limits
            time.sleep(1)
    
    # Final save
    print(f"\n\n{'='*70}")
    print("FINAL SAVE")
    print(f"{'='*70}")
    save_checkpoint(df_scored, output_file)
    
    # Summary statistics
    elapsed_total = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("SCORING COMPLETE")
    print(f"{'='*70}")
    print(f"\nProcessed in this session: {processed_in_session:,}")
    print(f"Total time: {elapsed_total/60:.1f} minutes")
    print(f"Average rate: {processed_in_session/elapsed_total:.2f} conversations/second")
    print(f"Failed scores: {failed_count}")
    
    # Score statistics
    valid_empathy = df_scored['empathy_score'].notna().sum()
    valid_attachment = df_scored['attachment_score'].notna().sum()
    
    print(f"\nFinal Statistics:")
    print(f"  Total rows: {len(df_scored):,}")
    print(f"  Valid empathy scores: {valid_empathy:,} ({valid_empathy/len(df_scored)*100:.1f}%)")
    print(f"  Valid attachment scores: {valid_attachment:,} ({valid_attachment/len(df_scored)*100:.1f}%)")
    print(f"\nOutput saved to: {output_file}")
    print(f"{'='*70}\n")
    
    return df_scored


def main():
    parser = argparse.ArgumentParser(
        description='Score conversations incrementally with automatic checkpointing (sequential processing)'
    )
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--chunk-size', type=int, default=1000, 
                        help='Save progress every N conversations (default: 1000)')
    parser.add_argument('--start-from', type=int, default=None,
                        help='Start from specific row number (overrides auto-resume)')
    
    args = parser.parse_args()
    
    # Run incremental scoring
    score_conversations_incremental(
        args.input,
        args.output,
        chunk_size=args.chunk_size,
        start_from=args.start_from
    )


if __name__ == '__main__':
    main()
