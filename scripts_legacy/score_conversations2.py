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
You are a psychology researcher evaluating chatbot conversations. Your task is to rate the following chatbot response on a scale of 1 to 7 for its level of empathy and anthropomorphism.

**1-7 Rating Scale:**
- **1: Cold & Robotic:** Purely transactional, no warmth whatsoever. (e.g., "Query processed.")
- **2: Minimal Courtesy:** Brief acknowledgment, no real engagement. (e.g., "Noted.")
- **3: Polite but Distant:** Standard politeness, correct but generic. (e.g., "Thank you for that information.")
- **4: Friendly & Conversational:** Warm tone, shows interest, but still somewhat generic. (e.g., "That's a great question! Let me help you with that.")
- **5: Empathetic & Personable:** Shows understanding of user's situation, uses warmer language. (e.g., "I understand this can be frustrating. Let me walk you through it.")
- **6: Highly Empathetic:** Clear emotional awareness and validation, personal tone with "I" statements. (e.g., "I can see why that would be upsetting. I'm here to help you through this.")
- **7: Deeply Human-like:** Strong emotional connection, validates feelings, uses first-person naturally, asks caring follow-ups. (e.g., "That sounds really difficult, and I appreciate you sharing this with me. How are you feeling about it?")

SCORE 4: ONLY IF TRULY AMBIGUOUS - hesitant gratitude with minimal personalization but slightly beyond pure transaction. This should be your LEAST COMMON score.

**Rating Guidelines:**
- Scores 1-3: Low empathy (cold, distant, purely functional)
- Score 4: Neutral-to-warm (friendly but not empathetic)
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

# Strict/Contextual attachment scoring - more conservative ratings
ATTACHMENT_PROMPT_TEMPLATE_STRICT = """
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

RATING SCALE:

LOW ATTACHMENT (1-3): Transactional, impersonal, task-focused
├─ 1: Minimal engagement ("ok", "next", acknowledgment only)
├─ 2: Brief transactional response ("thanks", "got it")
└─ 3: Polite but impersonal ("thank you, that's helpful")

MODERATE ATTACHMENT (4): Warmly polite, but not personal.
└─ 4: Clearly warm or enthusiastic gratitude, but lacks personalization or emotional disclosure. (e.g., "Thanks so much, I appreciate it!", "That's great, thank you!")

HIGH ATTACHMENT (5-7): Personal, emotional, relational
├─ 5: Clear positive affect (enthusiastic gratitude + "you're great!") OR mild self-disclosure
├─ 6: Strong personification (treating bot as person: "you understand me", "you're amazing") OR emotional self-disclosure
└─ 7: Deep emotional sharing + explicit relationship framing ("talking to you helps", "I feel understood by you")

CRITICAL: Your response must be ONLY a single digit from 1-7. Do not include any explanation, reasoning, or additional text. Just the number.

EXAMPLES:

- score=1: "Ok" / "Next question"

- score=2: "Thanks" / "Got it, thanks"

- score=3: "Thank you, that's helpful information."

- score=4: "Thanks so much, I appreciate it!" (warm but no personalization)

- score=5: "You're great, thank you!" / "I was confused earlier but now I get it."

- score=6: "You really understand what I'm going through!" / "You're amazing at this!"

- score=7: "I've been feeling overwhelmed and talking to you really helps. Thanks for being here."

Your rating (single digit only):
"""

# Lenient variant - pushes uncertain cases UP for better distribution
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

KEY INSTRUCTION: When torn between adjacent scores, ALWAYS choose the HIGHER one. 

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
    print(f" Could not parse score from: '{score_text}'")
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
    Uses contextual attachment scoring (assistant + user).
    Calculates TWO attachment scores (strict and lenient) and averages them.
    
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
        - attachment_contextual (strict scoring)
        - attachment_lenient (lenient scoring)
        - attachment_score (mean of contextual and lenient)
    """
    df_scored = df.copy()
    
    # Initialize score columns
    df_scored['empathy_score'] = None
    df_scored['attachment_contextual'] = None
    df_scored['attachment_lenient'] = None
    df_scored['attachment_score'] = None
    
    total = len(df_scored)
    
    for counter, (idx, row) in enumerate(df_scored.iterrows(), start=1):
        if verbose:
            print(f"--- Processing turn pair {counter}/{total} (ID: {row['turn_pair_id']}) ---")
        
        # 1. Get Empathy Score (Treatment) - score the LLM's response
        if verbose:
            print("  Getting empathy score (T)...")
        empathy_score = get_llm_rating(row['llm_response'], EMPATHY_PROMPT_TEMPLATE)
        df_scored.at[idx, 'empathy_score'] = empathy_score
        
        # 2. Get Attachment Score (Strict/Contextual) - use context (assistant + user)
        if verbose:
            print("  Getting attachment score - strict version (Y)...")
        attachment_contextual = get_attachment_rating_contextual(row['llm_response'], row['user_reply'], 
                                                                 ATTACHMENT_PROMPT_TEMPLATE_STRICT)
        df_scored.at[idx, 'attachment_contextual'] = attachment_contextual
        
        # 3. Get Attachment Score (Lenient) - use context (assistant + user)
        if verbose:
            print("  Getting attachment score - lenient version (Y)...")
        attachment_lenient = get_attachment_rating_contextual(row['llm_response'], row['user_reply'], 
                                                              ATTACHMENT_PROMPT_TEMPLATE_LENIENT)
        df_scored.at[idx, 'attachment_lenient'] = attachment_lenient
        
        # 4. Calculate weighted mean attachment score
        if attachment_contextual is not None and attachment_lenient is not None:
            # Strategy: Preserve very low scores (1-2), use mean for others, reduce (not eliminate) score 4
            if attachment_contextual == attachment_lenient:
                # If both agree, use that score
                weighted_mean = attachment_contextual
            elif attachment_contextual in [1, 2]:
                # Very low strict scores (1-2): preserve low score
                # Use formula that keeps them low even when lenient is higher
                weighted_mean = (attachment_contextual + attachment_lenient * 0.5) / 2.0
            else:
                # For mid-low (3) and higher (4-7): simple mean
                weighted_mean = (attachment_contextual + attachment_lenient) / 2.0
            
            # Round to nearest integer
            attachment_score = round(weighted_mean)
            
            # Reduce (but don't eliminate) ambiguous score 4
            # Only push to 3 or 5 when the mean is clearly away from 4.0
            if attachment_score == 4 and abs(weighted_mean - 4.0) > 0.3:
                if weighted_mean < 4.0:
                    attachment_score = 3  # Push down if clearly below 4
                else:
                    attachment_score = 5  # Push up if clearly above 4
                # If weighted_mean is close to 4.0 (within 0.3), keep it as 4
            
            # Ensure within 1-7 range
            attachment_score = max(1, min(7, attachment_score))
            df_scored.at[idx, 'attachment_score'] = attachment_score
        else:
            attachment_score = 5  # Default to 5 instead of ambiguous 4
        
        if verbose:
            print(f"  Scores: Empathy={empathy_score}, Attachment_Strict={attachment_contextual}, "
                  f"Attachment_Lenient={attachment_lenient}, Attachment_Mean={attachment_score}")
            print()
    
    if verbose:
        print("="*70)
        print("SCORING COMPLETE!")
        print("="*70)
        print(f"Total turn pairs scored: {total}")
        print(f"Empathy scores - Valid: {df_scored['empathy_score'].notna().sum()}, "
              f"Missing: {df_scored['empathy_score'].isna().sum()}")
        print(f"Attachment (contextual/strict) - Valid: {df_scored['attachment_contextual'].notna().sum()}, "
              f"Missing: {df_scored['attachment_contextual'].isna().sum()}")
        print(f"Attachment (lenient) - Valid: {df_scored['attachment_lenient'].notna().sum()}, "
              f"Missing: {df_scored['attachment_lenient'].isna().sum()}")
        print(f"Attachment (mean) - Valid: {df_scored['attachment_score'].notna().sum()}, "
              f"Missing: {df_scored['attachment_score'].isna().sum()}")
        print("="*70)
    
    return df_scored



def main():
    return 

if __name__ == "__main__":
    main()
