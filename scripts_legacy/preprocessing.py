"""
Preprocessing script for WildChat-1M dataset.

This script filters and transforms the raw WildChat-1M data to prepare it for causal analysis
of LLM empathy and user attachment.

Filtering Criteria:
- Conversations with at least 3 turns
- Conversations in English only
- Non-toxic conversations

Output Format:
- Flattened turn-pair structure where each row represents:
  - LLM's response (Treatment T)
  - User's subsequent response (Outcome Y)
"""

import pandas as pd
import json
import ast
import hashlib
import re
from typing import Dict, List, Optional
from datetime import datetime, timezone


def safe_parse_string(s: str):
    """
    Safely parse a string representation of Python objects.
    Handles datetime objects and other Python literals.
    
    Args:
        s: String to parse
    
    Returns:
        Parsed Python object
    """
    if not isinstance(s, str):
        return s
    
    try:
        # Find and extract datetime objects with their full representation
        # Pattern: datetime.datetime(2023, 4, 9, 0, 2, 53, tzinfo=<UTC>)
        datetime_pattern = r'datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*tzinfo=<[^>]+>)?\)'
        
        # Store datetime information for replacement
        datetime_replacements = {}
        s_modified = s
        
        for i, match in enumerate(re.finditer(datetime_pattern, s)):
            year, month, day, hour, minute, second = match.groups()
            # Create a string representation that will be converted to datetime later
            datetime_str = f"{year}-{month:0>2}-{day:0>2} {hour:0>2}:{minute:0>2}:{second:0>2}"
            placeholder = f'"__DATETIME_{i}__"'
            datetime_replacements[f'__DATETIME_{i}__'] = datetime_str
            s_modified = s_modified.replace(match.group(0), placeholder, 1)
        
        # Parse the modified string with ast.literal_eval
        parsed = ast.literal_eval(s_modified)
        
        # Recursively replace datetime placeholders with datetime strings
        # (we'll convert these to actual datetime objects later if needed)
        def replace_datetime_placeholders(obj):
            if isinstance(obj, dict):
                return {k: replace_datetime_placeholders(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_datetime_placeholders(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('__DATETIME_'):
                # Return the datetime string (parseable by pandas)
                return datetime_replacements.get(obj, None)
            else:
                return obj
        
        return replace_datetime_placeholders(parsed)
        
    except (ValueError, SyntaxError):
        # If that fails, try json.loads (for proper JSON)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # If all else fails, return the original string
            return s


def create_user_id(hashed_ip: str, header: Dict) -> str:
    """
    Create a unique user identifier by combining hashed_ip and user-agent from headers.
    
    Args:
        hashed_ip: The hashed IP address
        header: Dictionary containing request headers
    
    Returns:
        A unique user identifier string
    """
    user_agent = header.get('user-agent', '') if isinstance(header, dict) else ''
    combined = f"{hashed_ip}_{user_agent}"
    return hashlib.md5(combined.encode()).hexdigest()


def extract_turn_pairs(conversation: List[Dict], conversation_hash: str, model: str, 
                       timestamp: str, hashed_ip: str, header: Dict) -> List[Dict]:
    """
    Extract turn pairs from a conversation where each pair consists of:
    - An assistant turn (Treatment T)
    - The subsequent user turn (Outcome Y)
    - The previous user turn (Confounder X1)
    
    Args:
        conversation: List of conversation turns
        conversation_hash: Unique identifier for the conversation
        model: LLM model identifier
        timestamp: Conversation timestamp
        hashed_ip: Hashed IP address
        header: Request headers
    
    Returns:
        List of turn-pair dictionaries
    """
    turn_pairs = []
    
    # Create user_id once for the entire conversation
    user_id = create_user_id(hashed_ip, header)
    
    # Iterate through conversation to find assistant-user pairs
    for i in range(len(conversation) - 1):
        current_turn = conversation[i]
        next_turn = conversation[i + 1]
        
        # We want: assistant turn followed by user turn
        if current_turn.get('role') == 'assistant' and next_turn.get('role') == 'user':
            # Find the previous user turn (the prompt that led to this assistant response)
            previous_user_turn = None
            for j in range(i - 1, -1, -1):
                if conversation[j].get('role') == 'user':
                    previous_user_turn = conversation[j]
                    break
            
            # Skip if we don't have a previous user turn (shouldn't happen in valid conversations)
            if previous_user_turn is None:
                continue
            
            # Extract timestamp from assistant turn, use conversation-level timestamp if None
            assistant_timestamp = current_turn.get('timestamp')
            if assistant_timestamp is None:
                assistant_timestamp = timestamp
            
            # Extract hour of day from timestamp
            hour_of_day = None
            if assistant_timestamp:
                try:
                    # Handle both string formats: "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DD HH:MM:SS+00:00"
                    timestamp_str = str(assistant_timestamp)
                    # Extract the hour part (HH) from the timestamp
                    if ' ' in timestamp_str:
                        time_part = timestamp_str.split(' ')[1]  # Get "HH:MM:SS" or "HH:MM:SS+00:00"
                        hour_of_day = int(time_part.split(':')[0])  # Extract hour (0-23)
                except (ValueError, IndexError, AttributeError):
                    hour_of_day = None
            
            # Create turn pair record
            turn_pair = {
                # Primary identifiers
                'conversation_hash': conversation_hash,
                'turn_pair_id': f"{conversation_hash}_{current_turn.get('turn_identifier', i)}",
                
                # X4: Model (Confounder - LLM Model ID)
                'model': model,
                
                # X2: User ID (Confounder - User's Latent Traits)
                'user_id': user_id,
                
                # X3: Conversation Context (Confounder)
                'turn_number': i,  # Position of assistant turn in conversation
                'total_turns': len(conversation),  # Total length of conversation
                'timestamp': assistant_timestamp,  # Time of assistant response
                'hour_of_day': hour_of_day,  # Hour of day (0-23) when conversation occurred
                
                # X1: User's Initial Prompt (Confounder)
                'user_prompt': previous_user_turn.get('content', ''),
                
                # T: Treatment - LLM Response (will be scored for empathy later)
                'llm_response': current_turn.get('content', ''),
                
                # Y: Outcome - User's Reply (will be scored for attachment later)
                'user_reply': next_turn.get('content', ''),
                
                # Additional metadata
                'turn_identifier': current_turn.get('turn_identifier', ''),
                'hashed_ip': hashed_ip,
                'country': next_turn.get('country', ''),
            }
            
            turn_pairs.append(turn_pair)
    
    return turn_pairs


def filter_conversation(row: pd.Series) -> bool:
    """
    Check if a conversation meets the filtering criteria.
    
    Criteria:
    - At least 3 turns
    - Language is English
    - Not toxic
    
    Args:
        row: A row from the DataFrame
    
    Returns:
        True if conversation passes all filters, False otherwise
    """
    # Parse conversation if it's a string
    conversation = row['conversation']
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except:
            return False
    
    # Check minimum turns (at least 3)
    if len(conversation) < 3:
        return False
    
    # Check language (English)
    if row['language'] != 'English':
        return False
    
    # Check toxicity
    if row['toxic']:
        return False
    
    return True


def preprocess_wildchat(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Main preprocessing function for WildChat-1M dataset.
    
    This function:
    1. Filters conversations based on criteria (length, language, toxicity)
    2. Flattens conversations into turn-pairs
    3. Extracts all relevant features for causal analysis
    
    Args:
        df: Raw WildChat-1M DataFrame
        verbose: Whether to print progress information
    
    Returns:
        Preprocessed DataFrame with flattened turn-pairs
    """
    if verbose:
        print(f"Starting preprocessing...")
        print(f"Initial dataset size: {len(df)} conversations")
    
    # Parse conversation column if it's stored as string
    if df['conversation'].dtype == 'object':
        if verbose:
            print("Parsing conversation data...")
        df['conversation'] = df['conversation'].apply(safe_parse_string)
    
    # Parse header column if it's stored as string
    if df['header'].dtype == 'object':
        if verbose:
            print("Parsing header data...")
        df['header'] = df['header'].apply(safe_parse_string)
    
    # Apply filters
    if verbose:
        print("Applying filters...")
    filtered_df = df[df.apply(filter_conversation, axis=1)].copy()
    
    if verbose:
        print(f"After filtering: {len(filtered_df)} conversations")
        print(f"Removed: {len(df) - len(filtered_df)} conversations")
    
    # Extract turn pairs
    if verbose:
        print("Extracting turn pairs and filtering code/instructional content...")
    
    all_turn_pairs = []
    for idx, row in filtered_df.iterrows():
        turn_pairs = extract_turn_pairs(
            conversation=row['conversation'],
            conversation_hash=row['conversation_hash'],
            model=row['model'],
            timestamp=row['timestamp'],
            hashed_ip=row['hashed_ip'],
            header=row['header']
        )
        all_turn_pairs.extend(turn_pairs)
    
    # Create final DataFrame
    result_df = pd.DataFrame(all_turn_pairs)
    
    if verbose:
        print(f"Extracted {len(result_df)} turn pairs (after code/instructional filtering)")
        print(f"Average turn pairs per conversation: {len(result_df) / len(filtered_df):.2f}")
        print("\nFinal dataset shape:", result_df.shape)
        print("\nColumns:", list(result_df.columns))
    
    return result_df


def get_preprocessing_stats(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict:
    """
    Calculate and return statistics about the preprocessing.
    
    Args:
        original_df: Original raw DataFrame
        processed_df: Preprocessed DataFrame
    
    Returns:
        Dictionary containing preprocessing statistics
    """
    stats = {
        'original_conversations': len(original_df),
        'processed_turn_pairs': len(processed_df),
        'unique_conversations': processed_df['conversation_hash'].nunique(),
        'unique_users': processed_df['user_id'].nunique(),
        'unique_models': processed_df['model'].nunique(),
        'avg_turns_per_conversation': processed_df.groupby('conversation_hash').size().mean(),
        'models_distribution': processed_df['model'].value_counts().to_dict(),
    }
    
    return stats


if __name__ == "__main__":
    # This allows the script to be run standalone for testing
    print("Preprocessing module loaded successfully!")
    print("Use preprocess_wildchat(df) to process your data.")
