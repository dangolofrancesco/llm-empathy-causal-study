import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    # Read the CSV
    data_path = Path(__file__).parent.parent / 'data' / 'wildchat_social_only_5plus.csv'
    df = pd.read_csv(data_path, on_bad_lines='warn')

    # Filter out rows with missing user_hashed_ip or user_content
    df = df[df['user_hashed_ip'].notnull() & df['user_content'].notnull()]

    # Clean user_content to avoid CSV issues
    df['user_content'] = df['user_content'].astype(str).str.replace(',', ' ').str.replace('\n', ' ').str.replace('\r', ' ')

    # Group by user_hashed_ip and aggregate
    agg_df = df.groupby('user_hashed_ip').agg(
        conversation = ('user_content', lambda x: ' '.join(x)),
        num_turns = ('user_content', 'count'),
        num_conversations = ('conversation_hash', pd.Series.nunique)
    ).reset_index()

    # Add null trait columns
    agg_df['trait_openness'] = None
    agg_df['trait_consciousness'] = None
    agg_df['trait_extraversion'] = None
    agg_df['trait agreableness'] = None
    agg_df['trait_neuroticism'] = None

    # Save to CSV
    output_csv = Path(__file__).parent.parent / 'data' / 'user_aggregate_conversations.csv'
    agg_df.to_csv(output_csv, index=False)
    print(f"Saved aggregated user DataFrame to: {output_csv}")
    print(agg_df.head())
