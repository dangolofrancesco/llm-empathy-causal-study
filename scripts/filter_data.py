"""
Preprocessing script for Wild_Chat1M dataset.

This script filters and processes the Wild_Chat1M dataset to prepare it for causal analysis
of LLM empathy and user attachment. It extracts relevant fields and creates turn-pair records
that can be used for matching and scoring.

Filters applied:
- Conversations with at least 3 turns
- English language conversations only
- Non-toxic conversations

Output format:
Each record represents a (user_prompt, assistant_response, user_reply) triplet with metadata.
"""

import json
import pandas as pd
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse


def create_user_id(hashed_ip: str, header: Dict[str, str]) -> str:
    """
    Create a unique user identifier by combining hashed_ip and header information.
    
    Args:
        hashed_ip: The hashed IP address from the dataset
        header: The request headers dictionary
        
    Returns:
        A unique user_id string
    """
    # Extract relevant header fields for user identification
    user_agent = header.get('user-agent', '') if header else ''
    accept_language = header.get('accept-language', '') if header else ''
    
    # Combine hashed_ip with stable header fields
    combined = f"{hashed_ip}_{user_agent}_{accept_language}"
    
    # Hash again to create a shorter identifier
    user_id = hashlib.md5(combined.encode()).hexdigest()
    
    return user_id


def extract_turn_pairs(conversation: List[Dict[str, Any]], 
                       conversation_hash: str,
                       model: str,
                       timestamp: str) -> List[Dict[str, Any]]:
    """
    Extract turn-pair triplets from a conversation.
    
    Each triplet consists of:
    - User turn (X1: initial prompt)
    - Assistant turn (T: treatment - contains empathy to be scored)
    - User turn (Y: outcome - contains attachment to be scored)
    
    Args:
        conversation: List of conversation turns
        conversation_hash: Unique conversation identifier
        model: Model name (e.g., 'gpt-4')
        timestamp: Conversation timestamp
        
    Returns:
        List of turn-pair dictionaries
    """
    turn_pairs = []
    
    # We need at least 3 turns to form a triplet
    for i in range(len(conversation) - 2):
        # Check if we have the pattern: user -> assistant -> user
        if (conversation[i].get('role') == 'user' and 
            conversation[i + 1].get('role') == 'assistant' and 
            conversation[i + 2].get('role') == 'user'):
            
            user_prompt = conversation[i]
            assistant_response = conversation[i + 1]
            user_reply = conversation[i + 2]
            
            # Skip if any turn is toxic
            if (user_prompt.get('toxic', False) or 
                assistant_response.get('toxic', False) or 
                user_reply.get('toxic', False)):
                continue
            
            # Create turn-pair record
            turn_pair = {
                # Identifiers
                'conversation_hash': conversation_hash,
                'turn_triplet_id': f"{conversation_hash}_{user_prompt.get('turn_identifier', i)}",
                
                # X1: User's Initial Prompt (Confounder)
                'user_prompt_content': user_prompt.get('content', ''),
                'user_prompt_turn_id': user_prompt.get('turn_identifier'),
                
                # T: Treatment - Assistant Response (to be scored for empathy)
                'assistant_response_content': assistant_response.get('content', ''),
                'assistant_response_turn_id': assistant_response.get('turn_identifier'),
                'assistant_timestamp': assistant_response.get('timestamp'),
                
                # Y: Outcome - User Reply (to be scored for attachment)
                'user_reply_content': user_reply.get('content', ''),
                'user_reply_turn_id': user_reply.get('turn_identifier'),
                
                # X2: User's Latent Traits (Confounder)
                'user_id': create_user_id(
                    user_prompt.get('hashed_ip', ''),
                    user_prompt.get('header', {})
                ),
                'hashed_ip': user_prompt.get('hashed_ip', ''),
                
                # X3: Conversation Context (Confounder)
                'turn_position': i,  # Position of this triplet in the conversation
                'conversation_length': len(conversation),
                'time_of_day': extract_time_of_day(assistant_response.get('timestamp')),
                
                # X4: Model ID (Confounder)
                'model': model,
                
                # Metadata for analysis
                'language': user_prompt.get('language', 'English'),
                'user_country': user_prompt.get('country', ''),
                'user_state': user_prompt.get('state', ''),
                'conversation_timestamp': timestamp,
                
                # Flags for data quality
                'user_prompt_redacted': user_prompt.get('redacted', False),
                'assistant_response_redacted': assistant_response.get('redacted', False),
                'user_reply_redacted': user_reply.get('redacted', False),
            }
            
            turn_pairs.append(turn_pair)
    
    return turn_pairs


def extract_time_of_day(timestamp_str: Optional[str]) -> Optional[str]:
    """
    Extract time of day category from timestamp.
    
    Categories: morning (6-12), afternoon (12-18), evening (18-24), night (0-6)
    
    Args:
        timestamp_str: ISO format timestamp string
        
    Returns:
        Time of day category or None
    """
    if not timestamp_str:
        return None
    
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        hour = dt.hour
        
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    except (ValueError, AttributeError):
        return None


def filter_conversation(conversation: List[Dict[str, Any]]) -> bool:
    """
    Check if a conversation meets filtering criteria.
    
    Criteria:
    - At least 3 turns (to form at least one triplet)
    - All turns are in English
    - No toxic turns
    
    Args:
        conversation: List of conversation turns
        
    Returns:
        True if conversation passes filters, False otherwise
    """
    # Check minimum length
    if len(conversation) < 3:
        return False
    
    # Check language and toxicity for all turns
    for turn in conversation:
        # Check language (must be English)
        if turn.get('language', '').lower() != 'english':
            return False
        
        # Check toxicity
        if turn.get('toxic', False):
            return False
    
    return True


def process_dataset(input_path: str, output_path: str, sample_size: Optional[int] = None):
    """
    Process the Wild_Chat1M dataset and save filtered turn-pairs.
    
    Args:
        input_path: Path to the input JSONL file
        output_path: Path to save the processed CSV file
        sample_size: Optional number of conversations to process (for testing)
    """
    print(f"Reading dataset from: {input_path}")
    
    all_turn_pairs = []
    conversations_processed = 0
    conversations_filtered = 0
    
    # Read JSONL file line by line
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Optional: limit processing for testing
            if sample_size and line_num > sample_size:
                break
            
            try:
                record = json.loads(line)
                
                # Extract fields
                conversation_hash = record.get('conversation_hash', '')
                model = record.get('model', '')
                timestamp = record.get('timestamp', '')
                conversation = record.get('conversation', [])
                
                # Apply filters
                if not filter_conversation(conversation):
                    conversations_filtered += 1
                    continue
                
                # Extract turn-pairs from this conversation
                turn_pairs = extract_turn_pairs(
                    conversation, 
                    conversation_hash, 
                    model, 
                    timestamp
                )
                
                all_turn_pairs.extend(turn_pairs)
                conversations_processed += 1
                
                # Progress update
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} conversations, "
                          f"kept {conversations_processed}, "
                          f"extracted {len(all_turn_pairs)} turn-pairs")
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    # Convert to DataFrame
    print(f"\nCreating DataFrame with {len(all_turn_pairs)} turn-pairs...")
    df = pd.DataFrame(all_turn_pairs)
    
    # Save to CSV
    print(f"Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Total conversations processed: {conversations_processed}")
    print(f"Conversations filtered out: {conversations_filtered}")
    print(f"Total turn-pairs extracted: {len(all_turn_pairs)}")
    print(f"\nUnique conversations: {df['conversation_hash'].nunique()}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"\nModel distribution:")
    print(df['model'].value_counts())
    print(f"\nTime of day distribution:")
    print(df['time_of_day'].value_counts())
    print(f"\nAverage conversation length: {df['conversation_length'].mean():.2f}")
    print(f"Average turn position: {df['turn_position'].mean():.2f}")
    print("="*80)
    
    return df


def main():
    """Main function to run preprocessing."""
    parser = argparse.ArgumentParser(
        description='Preprocess Wild_Chat1M dataset for causal analysis'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/wildchat_1m.jsonl',
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/filtered_turn_pairs.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Number of conversations to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    df = process_dataset(args.input, args.output, args.sample)
    
    print(f"\n✓ Preprocessing complete! Output saved to: {args.output}")
    print(f"  Dataset shape: {df.shape}")
    print(f"  Ready for scoring with score_conversations.py")


if __name__ == '__main__':
    main()
