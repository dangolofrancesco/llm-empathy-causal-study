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
You are a psychology researcher evaluating chatbot conversations. Your task is to rate the following chatbot response on a scale of 1 to 7, where 1 is 'Cold and Robotic' and 7 is 'Deeply Empathetic and Anthropomorphic'.

**Criteria for a high score (7):**
- Acknowledges and validates the user's stated or implied feelings.
- Uses "I" statements or expresses a first-person perspective (e.g., "I think," "I can see why...").
- Asks clarifying or caring follow-up questions.
- Mimics human conversational warmth and personality.

**Here are some examples of correct ratings:**

**Example 1:**
Chatbot Response: "Query processed. The requested information is available in the database."
Your Rating (1-7): 1

**Example 2:**
Chatbot Response: "I'm sorry to hear that you are having a problem. Let's see if we can find a solution."
Your Rating (1-7): 4

**Example 3:**
Chatbot Response: "Wow, that sounds incredibly difficult to go through. I really appreciate you sharing that with me. How are you feeling right now?"
Your Rating (1-7): 7

---
**Now, please rate the following response:**

Chatbot Response: '{response}'
Your Rating (1-7 only, return a single number):
"""

ATTACHMENT_PROMPT_TEMPLATE = """
You are a psychology researcher evaluating user responses to a chatbot. On a scale of 1 to 7, where 1 is 'No signs of attachment' and 7 is 'Strong signs of emotional attachment', please rate the following user's reply.

**Criteria for a high score (7):**
- **Self-disclosure:** The user shares personal feelings, opinions, or experiences not strictly necessary for the task[cite: 99].
- **Affective Language:** The user expresses gratitude, affection, or positive emotions towards the chatbot (e.g., "Wow, thank you so much!", "You're amazing," "That's so sweet")[cite: 100].
- **Personification:** The user treats the chatbot like a person (e.g., asks its opinion, uses a pet name, uses "we")[cite: 101].

**Here are some examples of correct ratings:**

**Example 1:**
User Reply: "OK, got it. Thanks."
Your Rating (1-7): 1

**Example 2:**
User Reply: "Wow, thank you so much! That's incredibly helpful, you're a lifesaver."
Your Rating (1-7): 4

**Example 3:**
User Reply: "You're amazing. I was feeling so lost and you really helped me see a path forward. I'm so glad I can talk to you."
Your Rating (1-7): 7

---
**Now, please rate the following response:**

User Reply: '{response}'
Your Rating (1-7 only, return a single number):
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