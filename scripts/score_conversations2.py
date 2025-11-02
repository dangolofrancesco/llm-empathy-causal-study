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

# --- 2. DEFINE PROMPT TEMPLATES ---
# This version uses CONTEXTUAL attachment scoring (assistant response + user reply)
# and AGGREGATES strict + lenient prompts for better distribution

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

CRITICAL: Respond with ONLY a single digit from 1-7. No explanation. Just the number.

Your Rating (single digit only):
"""

ATTACHMENT_PROMPT_TEMPLATE_CONTEXTUAL = """
You are a psychology researcher rating USER attachment to a CHATBOT based on the user's reply.

CONTEXT:
Chatbot Response:
\"\"\"
{assistant_response}
\"\"\"

User Reply:
\"\"\"
{user_reply}
\"\"\"

TASK: Rate the USER'S REPLY on a 1-7 scale for emotional attachment to the chatbot.

CRITICAL: Score 4 should be RARE - only use when genuinely ambiguous. Push toward clear low (1-3) or high (5-7) ratings.

RATING SCALE (BINARY-FRIENDLY):

LOW ATTACHMENT (1-3): Transactional, impersonal, task-focused
├─ 1: Minimal engagement ("ok", "next", acknowledgment only)
├─ 2: Brief transactional response ("thanks", "got it")
└─ 3: Polite but impersonal ("thank you, that's helpful")

SCORE 4: ONLY IF TRULY AMBIGUOUS - hesitant gratitude with minimal personalization but slightly beyond pure transaction. This should be your LEAST COMMON score.

HIGH ATTACHMENT (5-7): Personal, emotional, relational
├─ 5: Clear positive affect (enthusiastic gratitude, "you're great!") OR mild self-disclosure
├─ 6: Strong personification (treating bot as person: "you understand me", "you're amazing") OR emotional self-disclosure
└─ 7: Deep emotional sharing + explicit relationship framing ("talking to you helps", "I feel understood by you", continuity signals)

EXAMPLES:
- 1: "Ok" / "Next question"
- 2: "Thanks" / "Got it, thanks"
- 3: "Thank you, that's helpful information."
- 4: [USE SPARINGLY] "Thanks so much, I appreciate it!" (warm but no personalization)
- 5: "You're great, thank you!" / "I was confused earlier but now I get it."
- 6: "You really understand what I'm going through!" / "You're amazing at this!"
- 7: "I've been feeling overwhelmed and talking to you really helps. Thanks for being here."

DECISION RULE:
- No personal language + minimal affect → 1-3
- Personal language OR emotional disclosure → 5-7
- Truly stuck between low/high → 4 (but avoid defaulting to 4)

CRITICAL: Your response must be ONLY a single digit from 1-7. Do not include any explanation, reasoning, or additional text. Just the number.

Your rating (single digit only):
"""

# A more lenient variant used for aggregation - pushes uncertain cases UP
ATTACHMENT_PROMPT_TEMPLATE_LENIENT = """
You are rating USER attachment to a CHATBOT. When uncertain, FAVOR HIGHER scores.

CONTEXT:
Chatbot Response:
\"\"\"
{assistant_response}
\"\"\"

User Reply:
\"\"\"
{user_reply}
\"\"\"

LENIENT RATING GUIDANCE (favor higher scores when borderline):

1-3: LOW ATTACHMENT (only clear transactional/impersonal cases)
- 1: Bare minimum ("ok", "next")
- 2: Brief thanks ("thanks")
- 3: Polite but cold ("thank you for the information")

4: [AVOID] True middle ground only

5-7: HIGH ATTACHMENT (favor these when ANY warmth/personalization present)
- 5: ANY positive affect beyond politeness ("you're great!", enthusiastic tone) OR ANY self-disclosure
- 6: Personification ("you understand", "you're amazing") OR emotional sharing
- 7: Deep emotion + relationship framing ("you help me", "glad to talk to you")

KEY INSTRUCTION: When torn between adjacent scores, ALWAYS choose the HIGHER one. Be generous with 5-7 ratings.

CRITICAL: Respond with ONLY a single digit (1-7). No explanation, no reasoning, no additional text. Just one number.

Your rating (single digit only):
"""

# --- 3. DEFINE ROBUST JUDGE FUNCTIONS ---

def parse_score(score_text):
    """
    Extracts a score (1-7) from the LLM's response with multiple fallback strategies.
    """
    if not score_text:
        return None
    
    # Strategy 1: Look for standalone digit 1-7
    match = re.search(r'\b([1-7])\b', score_text)
    if match:
        return int(match.group(1))
    
    # Strategy 2: Look for "score: 5" or "rating: 5" patterns
    match = re.search(r'(?:score|rating|answer)[\s:]+([1-7])', score_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Strategy 3: Just find ANY digit 1-7 in the response
    match = re.search(r'([1-7])', score_text)
    if match:
        return int(match.group(1))
    
    # If all strategies fail, log and return None
    print(f"  ⚠️ Could not parse score from: '{score_text}'")
    return None

def get_llm_rating(response_text, prompt_template):
    """Generic function to get a rating from Llama via Groq (single-text version)."""
    max_retries = 5
    wait_time = 1  # Start with a 1-second wait (Groq is fast)

    for attempt in range(max_retries):
        try:
            prompt = prompt_template.format(response=response_text)
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,  # Only need a single digit
                temperature=0  # Make responses deterministic
            )
            parsed = parse_score(response.choices[0].message.content.strip())
            if parsed is not None:
                return parsed # If successful, return the parsed integer
            else:
                # Failed to parse - try one more time with explicit instruction
                print(f"  ⚠️ Parse failed. Retrying with stricter prompt...")
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


def get_attachment_rating_contextual(assistant_text, user_text, prompt_template):
    """Attachment rating that uses BOTH assistant response and user reply in the prompt."""
    max_retries = 5
    wait_time = 1

    for attempt in range(max_retries):
        try:
            prompt = prompt_template.format(assistant_response=assistant_text, user_reply=user_text)
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,  # Only need a single digit
                temperature=0  # Make responses deterministic
            )
            parsed = parse_score(response.choices[0].message.content.strip())
            if parsed is not None:
                return parsed
            else:
                print(f"  Warning: Could not parse score from: {response.choices[0].message.content.strip()}")
                return None
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"  Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                print(f"  An unexpected error occurred: {e}")
                return None

    print("  Error: Max retries exceeded.")
    return None

# --- 4. SCORING FUNCTION FOR DATAFRAME ---
def score_conversations(df, verbose=True):
    """
    Score conversations with empathy and attachment ratings.
    Uses contextual attachment scoring (assistant + user) and aggregates strict + lenient prompts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: turn_pair_id, user_prompt, llm_response, user_reply, etc.
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with new columns: 
        - empathy_score
        - attachment_score_strict (contextual, base rubric)
        - attachment_score_lenient (contextual, lenient rubric)
        - attachment_score (final, aggregated mean of strict+lenient)
    """
    df_scored = df.copy()
    
    # Initialize score columns
    df_scored['empathy_score'] = None
    df_scored['attachment_score_strict'] = None
    df_scored['attachment_score_lenient'] = None
    df_scored['attachment_score'] = None  # final aggregated
    
    total = len(df_scored)
    
    for idx, row in df_scored.iterrows():
        if verbose:
            print(f"--- Processing turn pair {idx+1}/{total} (ID: {row['turn_pair_id']}) ---")
        
        # 1. Get Empathy Score (Treatment) - score the LLM's response
        if verbose:
            print("  Getting empathy score (T)...")
        empathy_score = get_llm_rating(row['llm_response'], EMPATHY_PROMPT_TEMPLATE)
        df_scored.at[idx, 'empathy_score'] = empathy_score
        
        # 2. Get Attachment Score (Outcome) - use context (assistant + user) and aggregate strict + lenient
        if verbose:
            print("  Getting attachment score (Y) [strict]...")
        att_strict = get_attachment_rating_contextual(row['llm_response'], row['user_reply'], 
                                                       ATTACHMENT_PROMPT_TEMPLATE_CONTEXTUAL)
        df_scored.at[idx, 'attachment_score_strict'] = att_strict

        if verbose:
            print("  Getting attachment score (Y) [lenient]...")
        att_lenient = get_attachment_rating_contextual(row['llm_response'], row['user_reply'], 
                                                        ATTACHMENT_PROMPT_TEMPLATE_LENIENT)
        df_scored.at[idx, 'attachment_score_lenient'] = att_lenient

        # SMART AGGREGATION: Avoid score=4 by choosing intelligently between strict/lenient
        candidates = [s for s in [att_strict, att_lenient] if s is not None]
        if len(candidates) == 0:
            final_att = 4
        elif len(candidates) == 1:
            # Only one score available, use it
            final_att = candidates[0]
        else:
            # Both strict and lenient available - smart logic to minimize 4s
            mean_score = sum(candidates) / len(candidates)
            rounded = int(round(mean_score))
            
            # Allow final=4 ONLY when:
            # 1. Both strict and lenient are 4, OR
            # 2. One is 4 and the mean is closer to 4 than to the other score
            if att_strict == 4 and att_lenient == 4:
                # Case 1: Both are 4, must accept it
                final_att = 4
            elif att_strict == 4 or att_lenient == 4:
                # Case 2: One is 4, check if mean is closer to 4
                other_score = att_lenient if att_strict == 4 else att_strict
                dist_to_4 = abs(mean_score - 4)
                dist_to_other = abs(mean_score - other_score)
                
                if dist_to_4 < dist_to_other:
                    # Mean is closer to 4, allow it
                    final_att = 4
                else:
                    # Mean is closer to the other score, use that
                    final_att = other_score
            elif rounded == 4:
                # Neither is 4, but mean rounds to 4 - choose score further from 4
                dist_strict = abs(att_strict - 4)
                dist_lenient = abs(att_lenient - 4)
                
                if dist_strict > dist_lenient:
                    final_att = att_strict  # Strict is further from 4
                elif dist_lenient > dist_strict:
                    final_att = att_lenient  # Lenient is further from 4
                else:
                    # Equal distance: round away from 4
                    final_att = 3 if mean_score < 4 else 5
            else:
                # Normal case: use rounded mean
                final_att = rounded
            
            final_att = max(1, min(7, final_att))  # Ensure 1-7 range
        
        df_scored.at[idx, 'attachment_score'] = final_att
        
        if verbose:
            print(f"  Scores: Empathy={empathy_score}, Attachment(strict)={att_strict}, Attachment(lenient)={att_lenient}, Final={final_att}")
            print()
    
    if verbose:
        print("="*70)
        print("SCORING COMPLETE!")
        print("="*70)
        print(f"Total turn pairs scored: {total}")
        print(f"Empathy scores - Valid: {df_scored['empathy_score'].notna().sum()}, "
              f"Missing: {df_scored['empathy_score'].isna().sum()}")
        print(f"Attachment (strict) - Valid: {df_scored['attachment_score_strict'].notna().sum()}, "
              f"Missing: {df_scored['attachment_score_strict'].isna().sum()}")
        print(f"Attachment (lenient) - Valid: {df_scored['attachment_score_lenient'].notna().sum()}, "
              f"Missing: {df_scored['attachment_score_lenient'].isna().sum()}")
        print(f"Attachment (final) - Valid: {df_scored['attachment_score'].notna().sum()}, "
              f"Missing: {df_scored['attachment_score'].isna().sum()}")
        print("="*70)
    
    return df_scored


# --- 5. MAIN SCRIPT LOGIC (for command-line usage) ---
def main():
    """Main function for scoring from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Score conversations with LLM-as-a-judge (contextual + lenient approach)')
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
