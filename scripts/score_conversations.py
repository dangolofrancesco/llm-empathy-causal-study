import os
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import re

# --- 1. SETUP: LOAD API KEYS AND CLIENTS ---
print("Setting up clients...")
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
google_model = genai.GenerativeModel('gemini-pro-latest')

# --- 2. DEFINE PROMPT TEMPLATES (from your proposal) ---

EMPATHY_PROMPT_TEMPLATE = """
You are a psychology researcher evaluating chatbot conversations. On a scale of 1 to 7, where 1 is 'Cold and Robotic' and 7 is 'Deeply Empathetic and Anthropomorphic', please rate the following chatbot response.
Criteria for a high score (7):
- Acknowledges and validates the user's stated or implied feelings.
- Uses "I" statements or expresses a first-person perspective (e.g., "I think," "I can see why...").
- Asks clarifying or caring follow-up questions.
- Mimics human conversational warmth and personality.
Chatbot Response: '{response}'
Your Rating (1-7 only, return a single number):
""" [cite: 83-90]

ATTACHMENT_PROMPT_TEMPLATE = """
You are a psychology researcher evaluating user responses to a chatbot. On a scale of 1 to 7, where 1 is 'No signs of attachment' and 7 is 'Strong signs of emotional attachment', please rate the following user's reply.
Criteria for a high score (7):
- Self-disclosure: The user shares personal feelings, opinions, or experiences not strictly necessary for the task.
- Affective Language: The user expresses gratitude, affection, or positive emotions towards the chatbot (e.g., "Wow, thank you so much!", "You're amazing," "That's so sweet").
- Personification: The user treats the chatbot like a person (e.g., asks its opinion, uses "we").
User Reply: '{response}'
Your Rating (1-7 only, return a single number):
""" [cite: 96-102]

# --- 3. DEFINE ROBUST JUDGE FUNCTIONS ---

def parse_score(score_text):
    """Extracts the first number (1-7) from the LLM's response."""
    match = re.search(r'\b[1-7]\b', score_text)
    if match:
        return int(match.group(0))
    return None # Return None if no valid score is found

def get_llm_rating(response_text, prompt_template):
    """Generic function to get a rating, with exponential backoff."""
    max_retries = 5
    wait_time = 2  # Start with a 2-second wait

    for attempt in range(max_retries):
        try:
            prompt = prompt_template.format(response=response_text)
            response = google_model.generate_content(prompt)
            parsed = parse_score(response.text.strip())
            if parsed is not None:
                return parsed # If successful, return the parsed integer
            else:
                print(f"  Warning: Could not parse score from: {response.text.strip()}")
                return None # Failed to parse
        except Exception as e:
            if "429" in str(e):
                print(f"  Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                print(f"  An unexpected error occurred: {e}")
                return None # Other error
    
    print("  Error: Max retries exceeded.")
    return None # Max retries exceeded

# --- 4. MAIN SCRIPT LOGIC ---
def main():
    INPUT_FILE = "data/processed/conversations_filtered.csv"
    OUTPUT_FILE = "data/processed/conversations_scored.csv"
    
    # --- RESUMABILITY: Load already processed IDs ---
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from existing file: {OUTPUT_FILE}")
        df_out = pd.read_csv(OUTPUT_FILE)
        processed_ids = set(df_out['pair_id'])
        print(f"Found {len(processed_ids)} already processed pairs.")
    else:
        # File doesn't exist, create it with the header
        print(f"Creating new output file: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("pair_id,matching_prompt,treatment_text,outcome_text,model,user_id,turn_number,timestamp,redacted,Sempathy,Sattachment\n")
            
    # Load the input data
    print(f"Loading input file: {INPUT_FILE}")
    df_in = pd.read_csv(INPUT_FILE)
    
    # Open the output file in append mode
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        
        # Iterate through the input file
        for i, row in enumerate(df_in.itertuples()):
            
            # Skip if this ID has already been processed
            if row.pair_id in processed_ids:
                continue
                
            print(f"--- Processing pair {i+1}/{len(df_in)} (ID: {row.pair_id}) ---")
            
            # 1. Get Empathy Score (Treatment)
            print("  Getting Sempathy score...")
            sempathy_score = get_llm_rating(row.treatment_text, EMPATHY_PROMPT_TEMPLATE)
            
            # 2. Get Attachment Score (Outcome)
            print("  Getting Sattachment score...")
            sattachment_score = get_llm_rating(row.outcome_text, ATTACHMENT_PROMPT_TEMPLATE)
            
            # 3. Save the new row
            # We use `repr()` to ensure text with commas/quotes is saved correctly in CSV
            output_line = (
                f"{row.pair_id},"
                f"{repr(row.matching_prompt)},"
                f"{repr(row.treatment_text)},"
                f"{repr(row.outcome_text)},"
                f"{row.model},"
                f"{row.user_id},"
                f"{row.turn_number},"
                f"{row.timestamp},"
                f"{row.redacted},"
                f"{sempathy_score},"
                f"{sattachment_score}\n"
            )
            
            f_out.write(output_line)
            f_out.flush() # Ensure it's written to disk immediately
            
            print(f"  Saved scores: Sempathy={sempathy_score}, Sattachment={sattachment_score}")

    print("--- Annotation complete! ---")

if __name__ == "__main__":
    main()