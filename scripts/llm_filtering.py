import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import sys

# Configuration
INPUT_FILE = "wildchat_filtered.jsonl"  # READING JSONL NOW
OUTPUT_FILE = "wildchat_social_only.jsonl" # SAVING AS JSONL

# Labels
CANDIDATE_LABELS = [
    "coding and debugging",       
    "creative writing task",      
    "factual question answering", 
    "personal advice",            
    "emotional support",          
    "casual conversation",        
    "roleplay",                   
]
KEEP_LABELS = {"personal advice", "emotional support", "casual conversation", "roleplay"}

def filter_social_conversations():
    print(">>> Loading Semantic Classifier...")
    
    # GPU Setup
    device = -1
    if torch.backends.mps.is_available():
        device = 0 
        print("Using GPU: MPS (Mac Acceleration)")
    else:
        print("Using CPU")

    classifier = pipeline("zero-shot-classification", 
                          model="facebook/bart-large-mnli", 
                          device=device)

    print(f">>> Loading JSONL: {INPUT_FILE}...")
    try:
        # lines=True tells pandas to read it as JSON Lines
        df = pd.read_json(INPUT_FILE, lines=True)
    except ValueError as e:
        print(f"Error reading JSONL: {e}")
        return

    print(f"Loaded {len(df)} rows. checking structure...")
    
    # Verify the first row is actually a list, not a string
    sample_conv = df.iloc[0]['conversation']
    if isinstance(sample_conv, str):
        print("CRITICAL ERROR: Data is still a string. Did you run Step 1?")
        return
    else:
        print(f"✅ Data structure is correct (List of {len(sample_conv)} turns).")

    social_records = []
    
    print(f"Processing {len(df)} conversations...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Direct access! No parsing needed.
            conv_data = row['conversation']
            
            # Extract User Text
            user_messages = [t.get('content', '') for t in conv_data if t.get('role') == 'user']
            if not user_messages:
                continue

            # Truncate
            user_text = " ".join(user_messages[:3])[:1000]
            if not user_text.strip():
                continue

            # Classify
            result = classifier(user_text, candidate_labels=CANDIDATE_LABELS, multi_label=False)
            top_label = result['labels'][0]
            score = result['scores'][0]

            # Filter
            if top_label in KEEP_LABELS:
                # Store dictionary (pandas row to dict)
                record = row.to_dict()
                record['semantic_intent'] = top_label
                record['intent_score'] = score
                social_records.append(record)
                
        except Exception as e:
            continue

    # Save
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Original Count: {len(df)}")
    print(f"Social/Empathic Count: {len(social_records)}")
    
    if len(social_records) > 0:
        df_social = pd.DataFrame(social_records)
        df_social.to_json(OUTPUT_FILE, orient='records', lines=True)
        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("No conversations passed the filter.")

if __name__ == "__main__":
    filter_social_conversations()