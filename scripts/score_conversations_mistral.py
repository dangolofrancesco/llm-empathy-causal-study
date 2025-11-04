"""
Alternative API configuration for faster scoring.

This script uses Mistral AI which has much higher rate limits (200 RPM vs Groq's 30 RPM).

Setup:
1. Get API key from: https://console.mistral.ai/
2. Add to .env: MISTRAL_API_KEY=your_key_here
3. Run this script instead of score_conversations_incremental.py

Rate comparison:
- Groq free tier: 30 RPM → ~0.06-0.1 conv/s → 618 hours
- Mistral free tier: 200 RPM → ~1-1.5 conv/s → 24-40 hours
"""

import os
import sys
import time
import pandas as pd
from mistralai import Mistral
from dotenv import load_dotenv
import argparse
from pathlib import Path

# Load environment
load_dotenv()

# Initialize Mistral client
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Same prompts as before
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

AATTACHMENT_PROMPT_TEMPLATE = """
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


def score_single_text_mistral(prompt_template, text, score_type, max_retries=3):
    """Score using Mistral AI"""
    prompt = prompt_template.format(response=text)
    
    for attempt in range(max_retries):
        try:
            response = mistral_client.chat.complete(
                model="mistral-small-latest",  # Fast and cheap
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
                
        except Exception as e:
            # More aggressive backoff for rate limits
            wait_time = (2 ** attempt) * 5  # Increased from 3 to 5 for better rate limit handling
            if "rate_limit" in str(e).lower() or "429" in str(e):
                # Extra wait for rate limits
                wait_time = wait_time + 5  # Increased from 2 to 5
                print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
            else:
                print(f"⚠️  Error scoring {score_type}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                return None
    
    return None


def load_progress(output_file):
    """Load existing progress - optimized for large files"""
    if os.path.exists(output_file):
        try:
            # First, quickly check how many rows have scores by reading in chunks
            print("  Checking existing progress (this may take a moment)...")
            scored_count = 0
            chunk_size = 10000
            
            for chunk in pd.read_csv(output_file, chunksize=chunk_size):
                scored_count += chunk['empathy_score'].notna().sum()
            
            last_processed = scored_count - 1 if scored_count > 0 else -1
            print(f"✓ Found existing progress: {scored_count:,} conversations with scores")
            if scored_count > 0:
                print(f"  Will resume from row {last_processed + 1}")
            
            # Now load the full file (we need it anyway)
            df_scored = pd.read_csv(output_file)
            return df_scored, last_processed
        except Exception as e:
            print(f"⚠️  Could not load progress file: {e}")
            return None, -1
    else:
        print("ℹ️  No existing progress found. Starting from beginning.")
        return None, -1


def save_checkpoint(df_scored, output_file):
    """Save progress"""
    try:
        df_scored.to_csv(output_file, index=False)
        return True
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        return False


def score_conversations_mistral(input_file, output_file, chunk_size=1000, start_from=None):
    """Score using Mistral AI (much faster!)"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("MISTRAL AI SCORING - STARTING")
    print(f"{'='*70}")
    print(f"\n✨ Using Mistral AI API (200 RPM - 6x faster than Groq!)")
    print(f"\nLoading dataset: {input_file}")
    df_input = pd.read_csv(input_file)
    total_rows = len(df_input)
    print(f"Total conversations to score: {total_rows:,}")
    
    df_scored, last_processed = load_progress(output_file)
    
    if start_from is not None:
        start_idx = start_from
    elif df_scored is not None:
        start_idx = last_processed + 1
    else:
        start_idx = 0
        df_scored = df_input.copy()
        df_scored['empathy_score'] = None
        df_scored['attachment_score'] = None
    
    if start_idx == 0 and df_scored is None:
        df_scored = df_input.copy()
        df_scored['empathy_score'] = None
        df_scored['attachment_score'] = None
    
    if start_idx > 0 and df_scored is not None:
        if 'empathy_score' not in df_scored.columns:
            df_scored['empathy_score'] = None
        if 'attachment_score' not in df_scored.columns:
            df_scored['attachment_score'] = None
        
        if len(df_input) > len(df_scored):
            new_rows = df_input.iloc[len(df_scored):].copy()
            new_rows['empathy_score'] = None
            new_rows['attachment_score'] = None
            df_scored = pd.concat([df_scored, new_rows], ignore_index=True)
    
    remaining = total_rows - start_idx
    
    print(f"\n{'='*70}")
    print("PROCESSING PLAN - SAMPLING APPROACH")
    print(f"{'='*70}")
    print(f"Total rows:           {total_rows:,}")
    print(f"Already processed:    {start_idx:,}")
    print(f"Remaining:            {remaining:,}")
    print(f"Chunk size:           {chunk_size:,}")
    print(f"Expected rate:        ~0.33 conv/s (balanced for sampling)")
    print(f"Estimated time:       {remaining/(0.33*3600):.1f} hours")
    print(f"\n💡 Target: 10k-20k scored conversations for valid analysis")
    print(f"   Stop script when target is reached")
    print(f"{'='*70}\n")
    
    if remaining == 0:
        print("✅ All conversations already scored!")
        return df_scored
    
    processed_in_session = 0
    failed_count = 0
    start_time = time.time()
    
    for i in range(start_idx, total_rows):
        row = df_input.iloc[i]
        
        empathy_score = score_single_text_mistral(
            EMPATHY_PROMPT_TEMPLATE, 
            row['llm_response'], 
            'empathy'
        )
        
        # Conservative delay between API calls to manage rate limits
        # For sampling approach: aim for steady progress, tolerate occasional rate limits
        # 1.5s between calls for balance between speed and stability
        time.sleep(1.5)
        
        attachment_score = score_single_text_mistral(
            ATTACHMENT_PROMPT_TEMPLATE,
            row['user_reply'],
            'attachment'
        )
        
        # Another delay after each conversation (total ~3.0s per conv = 0.33 conv/s)
        # Balanced for sampling: faster than 0.2 but more stable than 0.5
        time.sleep(1.5)
        
        df_scored.at[i, 'empathy_score'] = empathy_score
        df_scored.at[i, 'attachment_score'] = attachment_score
        
        if empathy_score is None or attachment_score is None:
            failed_count += 1
        
        processed_in_session += 1
        
        if processed_in_session % 10 == 0:
            progress = (processed_in_session / remaining) * 100
            elapsed = time.time() - start_time
            rate = processed_in_session / elapsed if elapsed > 0 else 0
            eta_seconds = (remaining - processed_in_session) / rate if rate > 0 else 0
            eta_hours = eta_seconds / 3600
            elapsed_hours = elapsed / 3600
            
            elapsed_str = f"{elapsed_hours:.1f}h" if elapsed_hours >= 1 else f"{elapsed/60:.1f}m"
            eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_seconds/60:.0f}m"
            
            print(f"Processing: {processed_in_session:,}/{remaining:,} ({progress:.1f}%) | "
                  f"Rate: {rate:.2f} conv/s | Elapsed: {elapsed_str} | ETA: {eta_str}", end='\r')
        
        if processed_in_session % chunk_size == 0:
            print(f"\n\n{'='*70}")
            print(f"CHECKPOINT - Saving progress...")
            print(f"{'='*70}")
            if save_checkpoint(df_scored, output_file):
                print(f"✓ Saved {i + 1:,} conversations")
                print(f"  Failed so far: {failed_count}")
                print(f"{'='*70}\n")
            time.sleep(0.5)
    
    print(f"\n\n{'='*70}")
    print("FINAL SAVE")
    print(f"{'='*70}")
    save_checkpoint(df_scored, output_file)
    
    elapsed_total = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("SCORING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nProcessed: {processed_in_session:,}")
    print(f"Total time: {elapsed_total/3600:.1f} hours")
    print(f"Average rate: {processed_in_session/elapsed_total:.2f} conv/s")
    print(f"Failed: {failed_count}")
    print(f"\nOutput: {output_file}")
    print(f"{'='*70}\n")
    
    return df_scored


def main():
    parser = argparse.ArgumentParser(description='Score using Mistral AI (faster!)')
    parser.add_argument('--input', required=True, help='Input CSV')
    parser.add_argument('--output', required=True, help='Output CSV')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--start-from', type=int, default=None)
    
    args = parser.parse_args()
    
    score_conversations_mistral(
        args.input,
        args.output,
        chunk_size=args.chunk_size,
        start_from=args.start_from
    )


if __name__ == '__main__':
    main()
