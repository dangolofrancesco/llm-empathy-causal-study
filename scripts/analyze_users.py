"""
Analyze user statistics in the WildChat dataset.

This script analyzes JSONL files to:
1. Count unique users (based on hashed_ip)
2. Count conversations per user
3. Provide summary statistics
4. Normalize hashed_ip values within conversations (use first IP for all turns)
5. Convert JSONL to CSV format for easier viewing

Usage:
    python3 analyze_users.py [filename]
    
If no filename is provided, defaults to wildchat_social_only.jsonl
"""

import json
import sys
import csv
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


def normalize_hashed_ips(input_path, output_path=None):
    """
    Normalize hashed_ip values within each conversation.
    
    For each conversation, all user turns will have their hashed_ip replaced
    with the first hashed_ip found in that conversation. This ensures
    consistency when the same user's IP changes mid-conversation.
    
    Args:
        input_path: Path to the input JSONL file
        output_path: Path to the output JSONL file (if None, will auto-generate)
        
    Returns:
        Path to the created normalized JSONL file
    """
    if output_path is None:
        # Auto-generate output path
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_normalized.jsonl"
    
    print(f"\nNormalizing hashed_ip values in {input_path.name}...")
    
    conversations_processed = 0
    conversations_normalized = 0
    ip_changes_found = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} conversations...")
            
            try:
                data = json.loads(line)
                conversation = data.get('conversation', [])
                
                # Find the first hashed_ip in user turns
                first_hashed_ip = None
                for turn in conversation:
                    if turn.get('role') == 'user':
                        hashed_ip = turn.get('hashed_ip')
                        if hashed_ip:
                            first_hashed_ip = hashed_ip
                            break
                
                # Normalize all user turns with the first hashed_ip
                if first_hashed_ip:
                    conversation_had_changes = False
                    for turn in conversation:
                        if turn.get('role') == 'user':
                            current_ip = turn.get('hashed_ip')
                            if current_ip and current_ip != first_hashed_ip:
                                conversation_had_changes = True
                                ip_changes_found += 1
                            # Set all user turns to the first hashed_ip
                            turn['hashed_ip'] = first_hashed_ip
                    
                    if conversation_had_changes:
                        conversations_normalized += 1
                
                # Write the normalized conversation
                outfile.write(json.dumps(data) + '\n')
                conversations_processed += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"\n✓ Normalization complete!")
    print(f"  Conversations processed: {conversations_processed:,}")
    print(f"  Conversations with IP changes: {conversations_normalized:,}")
    print(f"  Total IP changes normalized: {ip_changes_found:,}")
    print(f"  Output file: {output_path}")
    
    return output_path


def convert_jsonl_to_csv(jsonl_path, csv_path=None, max_rows=None):
    """
    Convert a JSONL file to CSV format for easier reading.
    
    Each turn (user-agent pair) becomes a row in the CSV.
    
    Args:
        jsonl_path: Path to the input JSONL file
        csv_path: Path to the output CSV file (if None, will auto-generate)
        max_rows: Maximum number of rows to convert (if None, convert all)
        
    Returns:
        Path to the created CSV file
    """
    if csv_path is None:
        # Auto-generate CSV path based on JSONL path
        csv_path = jsonl_path.with_suffix('.csv')
    
    print(f"Converting {jsonl_path.name} to CSV...")
    
    # Define CSV columns - one row per turn with user and agent columns
    fieldnames = [
        'conversation_hash',
        'conversation_model',
        'conversation_timestamp',
        'total_turns',
        'turn_number',
        'user_content',
        'user_hashed_ip',
        'user_country',
        'user_state',
        'user_language',
        'user_toxic',
        'user_redacted',
        'user_turn_identifier',
        'user_timestamp',
        'assistant_content',
        'assistant_language',
        'assistant_toxic',
        'assistant_redacted',
        'assistant_turn_identifier',
        'assistant_timestamp'
    ]
    
    rows_written = 0
    conversations_processed = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as infile, \
         open(csv_path, 'w', newline='', encoding='utf-8') as outfile:
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for line_num, line in enumerate(infile, 1):
            if max_rows and rows_written >= max_rows:
                print(f"  Reached maximum rows limit ({max_rows})")
                break
                
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} conversations, wrote {rows_written} rows...")
            
            try:
                data = json.loads(line)
                conversation_hash = data.get('conversation_hash', '')
                model = data.get('model', '')
                conv_timestamp = data.get('timestamp', '')
                conversation = data.get('conversation', [])
                
                # Count turns as user-agent pairs (count user messages)
                total_turns = sum(1 for msg in conversation if msg.get('role') == 'user')
                
                conversations_processed += 1
                
                # Pair up user and assistant messages into turns
                turn_number = 0
                i = 0
                while i < len(conversation):
                    msg = conversation[i]
                    
                    if msg.get('role') == 'user':
                        turn_number += 1
                        
                        # Initialize row with conversation metadata and user data
                        row = {
                            'conversation_hash': conversation_hash,
                            'conversation_model': model,
                            'conversation_timestamp': conv_timestamp,
                            'total_turns': total_turns,
                            'turn_number': turn_number,
                            'user_content': msg.get('content', '').replace('\n', ' '),
                            'user_hashed_ip': msg.get('hashed_ip', ''),
                            'user_country': msg.get('country', ''),
                            'user_state': msg.get('state', ''),
                            'user_language': msg.get('language', ''),
                            'user_toxic': msg.get('toxic', ''),
                            'user_redacted': msg.get('redacted', ''),
                            'user_turn_identifier': msg.get('turn_identifier', ''),
                            'user_timestamp': msg.get('timestamp', ''),
                            'assistant_content': '',
                            'assistant_language': '',
                            'assistant_toxic': '',
                            'assistant_redacted': '',
                            'assistant_turn_identifier': '',
                            'assistant_timestamp': ''
                        }
                        
                        # Look for the next assistant message
                        if i + 1 < len(conversation) and conversation[i + 1].get('role') == 'assistant':
                            assistant_msg = conversation[i + 1]
                            row['assistant_content'] = assistant_msg.get('content', '').replace('\n', ' ')
                            row['assistant_language'] = assistant_msg.get('language', '')
                            row['assistant_toxic'] = assistant_msg.get('toxic', '')
                            row['assistant_redacted'] = assistant_msg.get('redacted', '')
                            row['assistant_turn_identifier'] = assistant_msg.get('turn_identifier', '')
                            row['assistant_timestamp'] = assistant_msg.get('timestamp', '')
                            i += 1  # Skip the assistant message since we've processed it
                        
                        writer.writerow(row)
                        rows_written += 1
                        
                        if max_rows and rows_written >= max_rows:
                            break
                    
                    i += 1
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue
    
    print(f"\n✓ Conversion complete!")
    print(f"  Conversations processed: {conversations_processed:,}")
    print(f"  Rows written: {rows_written:,}")
    print(f"  Output file: {csv_path}")
    
    return csv_path


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
    
    # Option to normalize hashed_ip values
    normalize_option = input("\nWould you like to normalize hashed_ip values (use first IP for entire conversation)? (y/n): ")
    normalized_path = data_path
    if normalize_option.lower() == 'y':
        normalized_path = normalize_hashed_ips(data_path)
    
    # Option to convert to CSV
    csv_option = input("\nWould you like to convert the JSONL data to CSV format? (y/n): ")
    if csv_option.lower() == 'y':
        # Use normalized data if it was created, otherwise use original
        source_path = normalized_path if normalize_option.lower() == 'y' else data_path
        
        # Ask if they want to limit the number of rows
        limit_option = input("Convert all data? (y/n - 'n' to set a row limit): ")
        max_rows = None
        if limit_option.lower() == 'n':
            try:
                max_rows = int(input("Enter maximum number of rows to convert: "))
            except ValueError:
                print("Invalid number, converting all data.")
                max_rows = None
        
        convert_jsonl_to_csv(source_path, max_rows=max_rows)


if __name__ == '__main__':
    main()
