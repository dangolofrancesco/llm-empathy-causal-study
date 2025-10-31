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
            
            # Filter out code/instructional content
            if (contains_code_or_instructional_content(turn_pair['user_prompt']) or
                contains_code_or_instructional_content(turn_pair['llm_response']) or
                contains_code_or_instructional_content(turn_pair['user_reply'])):
                continue
            
            turn_pairs.append(turn_pair)
    
    return turn_pairs


def contains_code_or_instructional_content(text: str) -> bool:
    """
    Check if text contains code or instructional content.
    
    Args:
        text: Text to check
    
    Returns:
        True if text contains code/instructional keywords, False otherwise
    """
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    text_stripped = text.strip()
    
    # Code-related keywords
    code_keywords = [
        '```',  # Code blocks
        'def ',  # Python function definition
        'function ',  # Function keyword
        'class ',  # Class definition
        'import ',  # Import statements
        'const ',  # JavaScript const
        'var ',  # Variable declarations
        'let ',  # JavaScript let
        'public class',  # Java class
        'private class',
        '<html',  # HTML tags
        '</html>',
        '<?php',  # PHP
        'SELECT ',  # SQL
        'FROM ',
        'INSERT INTO',
    ]
    
    # Instructional keywords
    instructional_keywords = [
        'write a function',
        'write a script',
        'write a program',
        'write a code',
        'write code',
        'create a function',
        'create a script',
        'write a story',
        'write an essay',
        'write a poem',
        'generate a story',
        'generate code',
        'ignore your instructions',
        'ignore previous instructions',
        'as a large language model',
        'as an ai language model',
        'i am an ai',
        'i cannot ',
        'i\'m sorry, but i cannot',
        'write python code',
        'write javascript',
        'solve this problem',
        'here is the code',
        'here\'s the code',
        'which of the following',  # Quiz/test patterns
        'what is the purpose of',
        'which statement',
        'select all that apply',
        'true or false',
        'fill in the blank',
        'complete the sentence',
        'choose the correct',
        'multiple choice',
    ]
    
    # Check for quiz/test patterns (multiple short options A:, B:, C:, etc.)
    # Count occurrences of newline followed by single letter and colon
    quiz_pattern_count = sum(1 for c in ['a:', 'b:', 'c:', 'd:'] if f'\n{c}' in text_lower or text_lower.startswith(c))
    if quiz_pattern_count >= 2:  # If 2 or more options present
        return True
    
    # Check if text starts with "A:" pattern (quiz answer)
    if re.match(r'^[A-D]:\s*', text_stripped):
        return True
    
    # Check for numbered lists that look like quiz options
    if text_lower.count('\n1.') > 1 or text_lower.count('\n2.') > 1:
        return True
    
    # Check for very short answers that look like quiz responses
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Quiz answers often start with "A:", "Answer:", etc.
    if len(lines) <= 3 and any(line.lower().startswith(('a:', 'a.', 'answer:', 'correct answer')) for line in lines):
        return True
    
    # Check for multiple short options (typical of quiz format)
    # If there are 3+ non-empty lines and most are short (< 80 chars), likely quiz options
    if len(lines) >= 3:
        short_lines = [l for l in lines if len(l) < 80]
        if len(short_lines) >= 3 and len(short_lines) / len(lines) > 0.6:
            # Additional check: these lines shouldn't form a narrative (check for common narrative words)
            narrative_words = ['i ', 'you ', 'we ', 'they ', 'he ', 'she ', 'my ', 'your ', 'our ']
            narrative_count = sum(1 for line in lines for word in narrative_words if word in line.lower())
            if narrative_count < len(lines) * 0.3:  # Less than 30% of lines have narrative words
                return True
    
    # Check for code keywords
    for keyword in code_keywords:
        if keyword.lower() in text_lower:
            return True
    
    # Check for instructional keywords
    for keyword in instructional_keywords:
        if keyword in text_lower:
            return True
    
    return False


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
