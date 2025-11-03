import os
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import re

# --- 1. SETUP: LOAD API KEYS AND CLIENTS ---
print("Setting up clients...")
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- 2. DEFINE PROMPT TEMPLATES (from your proposal) ---

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

# --- 3. DEFINE ROBUST JUDGE FUNCTIONS ---

def parse_score(score_text):
    """Extracts the first number (1-7) from the LLM's response."""
    match = re.search(r'\b[1-7]\b', score_text)
    if match:
        return int(match.group(0))
    return None # Return None if no valid score is found

def get_llm_rating(response_text, prompt_template):
    """Generic function to get a rating from Llama via Groq, with exponential backoff."""
    max_retries = 5
    wait_time = 1  # Start with a 1-second wait (Groq is fast)

    for attempt in range(max_retries):
        try:
            prompt = prompt_template.format(response=response_text)
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10  # Only need a single number back
            )
            parsed = parse_score(response.choices[0].message.content.strip())
            if parsed is not None:
                return parsed # If successful, return the parsed integer
            else:
                print(f"  Warning: Could not parse score from: {response.choices[0].message.content.strip()}")
                return None # Failed to parse
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"  Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                print(f"  An unexpected error occurred: {e}")
                return None # Other error
    
    print("  Error: Max retries exceeded.")
    return None # Max retries exceeded

# --- 4. SCORING FUNCTION FOR DATAFRAME ---
def score_conversations(df, verbose=True):
    """
    Score conversations with empathy and attachment ratings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: turn_pair_id, user_prompt, llm_response, user_reply, etc.
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with two new columns: empathy_score, attachment_score
    """
    df_scored = df.copy()
    
    # Initialize score columns
    df_scored['empathy_score'] = None
    df_scored['attachment_score'] = None
    
    total = len(df_scored)
    
    for idx, row in df_scored.iterrows():
        if verbose:
            print(f"--- Processing turn pair {idx+1}/{total} (ID: {row['turn_pair_id']}) ---")
        
        # 1. Get Empathy Score (Treatment) - score the LLM's response
        if verbose:
            print("  Getting empathy score (T)...")
        empathy_score = get_llm_rating(row['llm_response'], EMPATHY_PROMPT_TEMPLATE)
        df_scored.at[idx, 'empathy_score'] = empathy_score
        
        # 2. Get Attachment Score (Outcome) - score the user's reply
        if verbose:
            print("  Getting attachment score (Y)...")
        attachment_score = get_llm_rating(row['user_reply'], ATTACHMENT_PROMPT_TEMPLATE)
        df_scored.at[idx, 'attachment_score'] = attachment_score
        
        if verbose:
            print(f"  Scores: Empathy={empathy_score}, Attachment={attachment_score}")
            print()
    
    if verbose:
        print("="*70)
        print("SCORING COMPLETE!")
        print("="*70)
        print(f"Total turn pairs scored: {total}")
        print(f"Empathy scores - Valid: {df_scored['empathy_score'].notna().sum()}, "
              f"Missing: {df_scored['empathy_score'].isna().sum()}")
        print(f"Attachment scores - Valid: {df_scored['attachment_score'].notna().sum()}, "
              f"Missing: {df_scored['attachment_score'].isna().sum()}")
        print("="*70)
    
    return df_scored


# --- 5. MAIN SCRIPT LOGIC (for command-line usage) ---
def main():
    """Main function for scoring from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Score conversations with LLM-as-a-judge')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')
    
    args = parser.parse_args()
    
    INPUT_FILE = args.input
    OUTPUT_FILE = args.output
    
    # Load input data
    print(f"Loading input file: {INPUT_FILE}")
    df_in = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df_in)} turn pairs")
    
    # Check if resuming
    if args.resume and os.path.exists(OUTPUT_FILE):
        print(f"\nResuming from existing file: {OUTPUT_FILE}")
        df_out = pd.read_csv(OUTPUT_FILE)
        processed_ids = set(df_out['turn_pair_id'])
        print(f"Found {len(processed_ids)} already processed pairs.")
        
        # Filter to only unprocessed rows
        df_to_process = df_in[~df_in['turn_pair_id'].isin(processed_ids)].reset_index(drop=True)
        print(f"Processing {len(df_to_process)} remaining pairs...")
        
        # Score the remaining pairs
        df_new_scored = score_conversations(df_to_process, verbose=True)
        
        # Combine with existing
        df_final = pd.concat([df_out, df_new_scored], ignore_index=True)
    else:
        # Score all pairs
        print("\nScoring all turn pairs...")
        df_final = score_conversations(df_in, verbose=True)
    
    # Save to output
    print(f"\nSaving scored data to: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved {len(df_final)} scored turn pairs")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()