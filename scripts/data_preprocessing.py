import pandas as pd
from datasets import load_dataset
from collections import Counter, defaultdict
import os


def print_user_distribution(user_conversation_counts, threshold_name, total_conversations):
    """
    Print detailed user distribution analysis similar to analyze_users.py
    
    Args:
        user_conversation_counts: Counter object with hashed_ip -> count
        threshold_name: Name of the threshold for display
        total_conversations: Total number of conversations
    """
    # Distribution of conversations per user
    conversations_per_user_dist = Counter(user_conversation_counts.values())
    unique_users = len(user_conversation_counts)
    
    print(f"\n{'─' * 80}")
    print(f"USER DISTRIBUTION FOR {threshold_name}")
    print(f"{'─' * 80}")
    
    print(f"\nTotal Conversations: {total_conversations:,}")
    print(f"Unique Users: {unique_users:,}")
    print(f"Average Conversations per User: {total_conversations / unique_users:.2f}")
    
    # Show distribution
    print(f"\n{'Conversations/User':<20} {'Number of Users':<20} {'% of Users':<15}")
    print("-" * 60)
    
    sorted_dist = sorted(conversations_per_user_dist.items())
    
    # Show first 20 entries
    display_limit = min(20, len(sorted_dist))
    for num_convs, num_users in sorted_dist[:display_limit]:
        pct = (num_users / unique_users) * 100
        print(f"{num_convs:<20} {num_users:<20,} {pct:>6.2f}%")
    
    if len(sorted_dist) > 20:
        print(f"... ({len(sorted_dist) - 20} more entries)\n")
        print("Users with most conversations:")
        for num_convs, num_users in sorted_dist[-5:]:
            pct = (num_users / unique_users) * 100
            print(f"{num_convs:<20} {num_users:<20,} {pct:>6.2f}%")
    
    # Calculate key metrics
    single_conv_users = conversations_per_user_dist.get(1, 0)
    multi_conv_users = unique_users - single_conv_users
    
    print(f"\nUsers with only 1 conversation: {single_conv_users:,} ({single_conv_users/unique_users*100:.2f}%)")
    print(f"Users with multiple conversations: {multi_conv_users:,} ({multi_conv_users/unique_users*100:.2f}%)")
    
    # Median
    all_conv_counts = []
    for num_convs, num_users in conversations_per_user_dist.items():
        all_conv_counts.extend([num_convs] * num_users)
    all_conv_counts.sort()
    median = all_conv_counts[len(all_conv_counts) // 2] if all_conv_counts else 0
    print(f"Median conversations per user: {median}")
    
    return {
        'unique_users': unique_users,
        'single_conv_users': single_conv_users,
        'multi_conv_users': multi_conv_users,
        'median': median,
        'distribution': conversations_per_user_dist
    }


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
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    total_rows = len(dataset)
    print(f"Total Conversations: {total_rows:,}")
    print(f"Columns: {dataset.column_names}")

    # --- 2. Initial Filtering (English only) ---
    print("\n" + "="*80)
    print("ENGLISH LANGUAGE FILTERING")
    print("="*80)
    
    # Filter 1: English Language
    english_dataset = dataset.filter(lambda x: x['language'] == 'English')
    eng_count = len(english_dataset)
    print(f"English Conversations: {eng_count:,} ({eng_count/total_rows:.1%})")
    print(f"Non-English Conversations: {total_rows - eng_count:,} ({(total_rows - eng_count)/total_rows:.1%})")

    # --- 3. Comprehensive User Analysis by Turn Threshold ---
    print("\n" + "="*80)
    print("USER STATISTICS ACROSS DIFFERENT TURN THRESHOLDS")
    print("="*80)
    
    # Analyze at different turn thresholds
    turn_thresholds = [3, 5, 8, 10, 12, 15]
    
    print(f"\n{'Threshold':<12} {'Conversations':<15} {'% of English':<15} {'Unique Users':<15} {'Avg Conv/User':<15}")
    print("-" * 80)
    
    threshold_results = {}
    for threshold in turn_thresholds:
        filtered = english_dataset.filter(lambda x: x['turn'] > threshold)
        conv_count = len(filtered)
        
        # Count conversations per user
        user_conversation_counts = Counter(filtered['hashed_ip'])
        unique_users = len(user_conversation_counts)
        avg_conv_per_user = conv_count / unique_users if unique_users > 0 else 0
        
        threshold_results[threshold] = {
            'conversations': conv_count,
            'unique_users': unique_users,
            'avg_conv_per_user': avg_conv_per_user,
            'dataset': filtered,
            'user_counts': user_conversation_counts
        }
        
        print(f">{threshold:<11} {conv_count:<15,} {conv_count/eng_count:<14.1%} {unique_users:<15,} {avg_conv_per_user:<15.2f}")
    
    # --- 4. Detailed User Distribution for Each Threshold ---
    print("\n" + "="*80)
    print("DETAILED USER DISTRIBUTIONS BY THRESHOLD")
    print("="*80)
    
    user_stats = {}
    for threshold in turn_thresholds:
        result = threshold_results[threshold]
        stats = print_user_distribution(
            result['user_counts'],
            f">{threshold} TURNS",
            result['conversations']
        )
        user_stats[threshold] = stats
    
    # --- 5. Turn Distribution Analysis ---
    print("\n" + "="*80)
    print("TURN DISTRIBUTION IN ENGLISH CONVERSATIONS")
    print("="*80)
    
    turn_counts = Counter(english_dataset['turn'])
    sorted_turns = sorted(turn_counts.items())
    
    print(f"\n{'Turns':<10} {'Conversations':<15} {'Cumulative':<15} {'% of Total':<15}")
    print("-" * 60)
    
    cumulative = 0
    for turns, count in sorted_turns[:20]:  # Show first 20
        cumulative += count
        pct_total = (count / eng_count) * 100
        pct_cumulative = (cumulative / eng_count) * 100
        print(f"{turns:<10} {count:<15,} {cumulative:<15,} {pct_total:>6.2f}% ({pct_cumulative:.1f}% cum)")
    
    if len(sorted_turns) > 20:
        remaining = sum(c for t, c in sorted_turns[20:])
        cumulative += remaining
        print(f"... (remaining {len(sorted_turns) - 20} turn counts)")
        print(f"{'TOTAL':<10} {eng_count:<15,} {cumulative:<15,} {100.0:>6.1f}%")

    # --- 6. Sample Inspection ---
    print("\n" + "="*80)
    print(f"SAMPLE ENTRY INSPECTION (>{turn_thresholds[-1]} turns)")
    print("="*80)
    
    # Use the most restrictive threshold for sample
    sample_threshold = turn_thresholds[-1]
    sample_dataset = threshold_results[sample_threshold]['dataset']
    sample_count = threshold_results[sample_threshold]['conversations']
    
    if sample_count > 0:
        sample = sample_dataset[0]
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
    
    # --- 7. Recommendations ---
    print("\n" + "="*80)
    print("RECOMMENDATIONS & CONCLUSION")
    print("="*80)
    
    print("\nBased on the analysis above, consider these turn thresholds:")
    
    # Show recommendations for available thresholds
    if 10 in threshold_results:
        print(f"\nOption 1 (Conservative, >10 turns):")
        print(f"  - Conversations: {threshold_results[10]['conversations']:,}")
        print(f"  - Unique Users: {threshold_results[10]['unique_users']:,}")
        print(f"  - Single-conv users: {user_stats[10]['single_conv_users']:,} ({user_stats[10]['single_conv_users']/threshold_results[10]['unique_users']*100:.1f}%)")
        print(f"  - Multi-conv users: {user_stats[10]['multi_conv_users']:,} ({user_stats[10]['multi_conv_users']/threshold_results[10]['unique_users']*100:.1f}%)")
        print(f"  - Trade-off: Very rich conversations, smaller dataset")
    
    if 5 in threshold_results:
        print(f"\nOption 2 (Moderate, >5 turns):")
        print(f"  - Conversations: {threshold_results[5]['conversations']:,}")
        print(f"  - Unique Users: {threshold_results[5]['unique_users']:,}")
        print(f"  - Single-conv users: {user_stats[5]['single_conv_users']:,} ({user_stats[5]['single_conv_users']/threshold_results[5]['unique_users']*100:.1f}%)")
        print(f"  - Multi-conv users: {user_stats[5]['multi_conv_users']:,} ({user_stats[5]['multi_conv_users']/threshold_results[5]['unique_users']*100:.1f}%)")
        print(f"  - Trade-off: Good balance between depth and sample size")
    
    if 3 in threshold_results:
        print(f"\nOption 3 (Liberal, >3 turns):")
        print(f"  - Conversations: {threshold_results[3]['conversations']:,}")
        print(f"  - Unique Users: {threshold_results[3]['unique_users']:,}")
        print(f"  - Single-conv users: {user_stats[3]['single_conv_users']:,} ({user_stats[3]['single_conv_users']/threshold_results[3]['unique_users']*100:.1f}%)")
        print(f"  - Multi-conv users: {user_stats[3]['multi_conv_users']:,} ({user_stats[3]['multi_conv_users']/threshold_results[3]['unique_users']*100:.1f}%)")
        print(f"  - Trade-off: Largest dataset, may include shallower interactions")
    
    # --- 8. Optional: Save filtered datasets ---
    print("\n" + "="*80)
    print("SAVE FILTERED DATASET?")
    print("="*80)
    
    threshold_options = '/'.join(str(t) for t in turn_thresholds) + '/skip'
    save_choice = input(f"\nWhich threshold would you like to save? ({threshold_options}): ").strip()
    
    if save_choice.isdigit() and int(save_choice) in threshold_results:
        threshold = int(save_choice)
        output_filename = f"wildchat_english_{threshold}plus_turns.jsonl"
        print(f"\nSaving dataset with >{threshold} turns to {output_filename} (JSONL format)...")
        print(f"This may take a few minutes...")
        
        # This saves native lists/dicts. No parsing needed later.
        threshold_results[threshold]['dataset'].to_json(output_filename)
        
        print(f"✓ Saved {threshold_results[threshold]['conversations']:,} conversations to {output_filename}")
        print(f"✓ Dataset contains {threshold_results[threshold]['unique_users']:,} unique users")
        print(f"✓ {user_stats[threshold]['single_conv_users']:,} users with 1 conversation")
        print(f"✓ {user_stats[threshold]['multi_conv_users']:,} users with multiple conversations")
    else:
        print("\nSkipping save. You can run this script again to save a filtered dataset.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    analyze_wildchat()