import pandas as pd
from datasets import load_dataset
import os

def analyze_wildchat():
    print(">>> Downloading and Loading WildChat-1M dataset...")
    # Load the dataset (split='train' usually contains the full data for this dataset)
    # Note: This might take a moment as it processes ~1M rows.
    try:
        dataset = load_dataset("allenai/WildChat-1M", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have the 'datasets' library installed (pip install datasets) and internet access.")
        return

    # --- 1. General Overview ---
    print("\n" + "="*40)
    print("DATASET OVERVIEW")
    print("="*40)
    total_rows = len(dataset)
    print(f"Total Conversations: {total_rows:,}")
    print(f"Columns: {dataset.column_names}")

    # --- 2. Initial Filtering (English & Length) ---
    print("\n" + "="*40)
    print("FILTERING STATISTICS")
    print("="*40)
    
    # Filter 1: English Language
    # The 'language' column indicates the dominant language.
    english_dataset = dataset.filter(lambda x: x['language'] == 'English')
    eng_count = len(english_dataset)
    print(f"English Conversations: {eng_count:,} ({eng_count/total_rows:.1%})")

    # Filter 2: Length (> 10 turns)
    # The 'turn' column allows us to filter for depth without counting the list manually.
    # Note: We filter specifically on the English subset.
    long_eng_dataset = english_dataset.filter(lambda x: x['turn'] > 10)
    long_count = len(long_eng_dataset)
    print(f"English & >10 Turns:   {long_count:,} ({long_count/total_rows:.1%})")

    # --- 3. User Grouping Potential ---
    # We check how many unique users exist in this filtered subset to support your "Personality Profiling" idea.
    # The 'hashed_ip' column is available at the top level.
    unique_users = len(set(long_eng_dataset['hashed_ip']))
    print(f"Unique Users (in filtered set): {unique_users:,}")
    print(f"Avg Conversations per User: {long_count / unique_users:.2f}")

    # --- 4. Sample Inspection ---
    print("\n" + "="*40)
    print("SAMPLE ENTRY INSPECTION")
    print("="*40)
    if long_count > 0:
        sample = long_eng_dataset[0]
        print(f"Conversation Hash: {sample['conversation_hash']}")
        print(f"Model: {sample['model']}")
        print(f"Turns: {sample['turn']}")
        print(f"Hashed IP: {sample['hashed_ip']}")
        
        # Verify conversation structure (User vs Assistant)
        print("\nFirst 2 Utterances:")
        for msg in sample['conversation'][:2]:
            role = msg.get('role', 'unknown').upper()
            content_preview = msg.get('content', '')[:100].replace('\n', ' ')
            print(f"[{role}]: {content_preview}...")
    
    print("\n" + "="*40)
    print("CONCLUSION FOR PROFESSOR")
    print("="*40)
    print(f"We can start the Semantic LLM Filtering process on a subset of {long_count:,} conversations.")
    print("These represent rich, multi-turn, English interactions suitable for latent personality profiling.")

    # --- 5. Saving to CSV ---
    print("\n" + "="*40)
    print("SAVING DATASET")
    print("="*40)
    
    # --- THE FIX: Save as JSONL ---
    output_filename = "wildchat_filtered.jsonl"
    print(f"Saving to {output_filename} (JSONL format)...")
    
    # This saves native lists/dicts. No parsing needed later.
    long_eng_dataset.to_json(output_filename)
    
    print("Done! Now run the filtering script.")

if __name__ == "__main__":
    analyze_wildchat()