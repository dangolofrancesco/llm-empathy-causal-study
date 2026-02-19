"""
Analyze user statistics in the WildChat dataset.

This script analyzes JSONL files to:
1. Count unique users (based on hashed_ip)
2. Count conversations per user
3. Provide summary statistics

Usage:
    python3 analyze_users.py [filename]
    
If no filename is provided, defaults to wildchat_social_only.jsonl
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyze_users(data_path):
    """
    Analyze user statistics from the WildChat dataset.
    
    Args:
        data_path: Path to the JSONL data file
        
    Returns:
        dict: Dictionary containing user statistics
    """
    user_conversations = defaultdict(list)  # user_ip -> list of conversation_hashes
    total_conversations = 0
    
    print(f"Reading data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} conversations...")
            
            try:
                data = json.loads(line)
                conversation_hash = data.get('conversation_hash')
                total_conversations += 1
                
                # Extract user IP from the first user message in the conversation
                conversation = data.get('conversation', [])
                for turn in conversation:
                    if turn.get('role') == 'user':
                        hashed_ip = turn.get('hashed_ip')
                        if hashed_ip:
                            user_conversations[hashed_ip].append(conversation_hash)
                            break  # Only need the first user message to identify the user
                            
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"  Processed {line_num} conversations total.")
    
    # Calculate statistics
    num_unique_users = len(user_conversations)
    conversations_per_user = [len(convs) for convs in user_conversations.values()]
    
    results = {
        'total_conversations': total_conversations,
        'unique_users': num_unique_users,
        'conversations_per_user': Counter(conversations_per_user),
        'avg_conversations_per_user': sum(conversations_per_user) / num_unique_users if num_unique_users > 0 else 0,
        'max_conversations_per_user': max(conversations_per_user) if conversations_per_user else 0,
        'min_conversations_per_user': min(conversations_per_user) if conversations_per_user else 0,
    }
    
    return results, user_conversations


def print_statistics(results):
    """Print formatted statistics."""
    print("\n" + "=" * 70)
    print("USER STATISTICS SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal conversations in dataset: {results['total_conversations']:,}")
    print(f"Unique users (by hashed_ip): {results['unique_users']:,}")
    print(f"\nAverage conversations per user: {results['avg_conversations_per_user']:.2f}")
    print(f"Minimum conversations per user: {results['min_conversations_per_user']}")
    print(f"Maximum conversations per user: {results['max_conversations_per_user']}")
    
    print("\n" + "-" * 70)
    print("DISTRIBUTION OF CONVERSATIONS PER USER")
    print("-" * 70)
    
    # Sort by number of conversations
    conv_distribution = sorted(results['conversations_per_user'].items())
    
    print(f"\n{'Conversations':<15} {'Users':<10} {'Percentage':<12}")
    print("-" * 40)
    
    total_users = results['unique_users']
    for num_convs, count in conv_distribution[:20]:  # Show first 20 entries
        percentage = (count / total_users) * 100
        print(f"{num_convs:<15} {count:<10,} {percentage:>6.2f}%")
    
    if len(conv_distribution) > 20:
        print(f"... ({len(conv_distribution) - 20} more entries)")
        # Show last few entries
        print("\nUsers with most conversations:")
        for num_convs, count in conv_distribution[-5:]:
            percentage = (count / total_users) * 100
            print(f"{num_convs:<15} {count:<10,} {percentage:>6.2f}%")
    
    # Additional statistics
    print("\n" + "-" * 70)
    print("ADDITIONAL INSIGHTS")
    print("-" * 70)
    
    # Count users with exactly 1 conversation
    single_conv_users = results['conversations_per_user'].get(1, 0)
    multi_conv_users = total_users - single_conv_users
    
    print(f"\nUsers with only 1 conversation: {single_conv_users:,} ({(single_conv_users/total_users)*100:.2f}%)")
    print(f"Users with multiple conversations: {multi_conv_users:,} ({(multi_conv_users/total_users)*100:.2f}%)")
    
    # Calculate median
    all_counts = []
    for num_convs, count in results['conversations_per_user'].items():
        all_counts.extend([num_convs] * count)
    all_counts.sort()
    median_idx = len(all_counts) // 2
    median = all_counts[median_idx] if all_counts else 0
    
    print(f"\nMedian conversations per user: {median}")
    
    print("\n" + "=" * 70)
    
    return median  # Return median for use in main


def main():
    """Main function to run the analysis."""
    # Check for command-line arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'wildchat_social_only.jsonl'
    
    # Define the data path
    data_path = Path(__file__).parent.parent / 'data' / filename
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print(f"\nAvailable files in data directory:")
        data_dir = Path(__file__).parent.parent / 'data'
        if data_dir.exists():
            for file in sorted(data_dir.glob('*.jsonl')):
                print(f"  - {file.name}")
        return
    
    # Run analysis
    results, user_conversations = analyze_users(data_path)
    
    # Print results, get median
    median = print_statistics(results)
    
    # Optionally save detailed results
    save_option = input("\nWould you like to save detailed user statistics to a file? (y/n): ")
    if save_option.lower() == 'y':
        # Create output filename based on input filename
        base_name = data_path.stem  # filename without extension
        output_filename = f'{base_name}_user_statistics.json'
        output_path = Path(__file__).parent.parent / 'data' / output_filename
        
        # Prepare data for JSON serialization
        output_data = {
            'source_file': filename,
            'summary': {
                'total_conversations': results['total_conversations'],
                'unique_users': results['unique_users'],
                'avg_conversations_per_user': results['avg_conversations_per_user'],
                'max_conversations_per_user': results['max_conversations_per_user'],
                'min_conversations_per_user': results['min_conversations_per_user'],
                'median_conversations_per_user': median,
            },
            'distribution': {str(k): v for k, v in results['conversations_per_user'].items()},
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDetailed statistics saved to: {output_path}")


if __name__ == '__main__':
    main()
