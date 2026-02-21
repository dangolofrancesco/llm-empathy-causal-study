import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    # Read the CSV
    data_path = Path(__file__).parent.parent / 'data' / 'wildchat_social_only_5plus.csv'
    df = pd.read_csv(data_path, on_bad_lines='warn')

    # Filter out rows with missing user_hashed_ip or user_content
    df = df[df['user_hashed_ip'].notnull() & df['user_content'].notnull()]
    df['user_content'] = df['user_content'].astype(str).str.replace(',', ' ').str.replace('\n', ' ').str.replace('\r', ' ')

    # Sort by user_hashed_ip and user_timestamp to ensure order
    df = df.sort_values(['user_hashed_ip', 'user_timestamp'])

    # Find last conversation for each user (for test set)
    last_convs = df.groupby('user_hashed_ip')['conversation_hash'].transform('max')
    is_last_conv = df['conversation_hash'] == last_convs

    # Test DataFrame: users with >1 conversation, only their last conversation's turns
    user_conv_counts = df.groupby('user_hashed_ip')['conversation_hash'].nunique()
    multi_conv_users = user_conv_counts[user_conv_counts > 1].index
    test_df = df[df['user_hashed_ip'].isin(multi_conv_users) & is_last_conv]
    test_agg = test_df.groupby('user_hashed_ip').agg(
        conversation = ('user_content', lambda x: ' '.join(x)),
        num_turns = ('user_content', 'count'),
        num_conversations = ('conversation_hash', pd.Series.nunique)
    ).reset_index()
    for col in ['trait_openness', 'trait_consciousness', 'trait_extraversion', 'trait agreableness', 'trait_neuroticism']:
        test_agg[col] = None

    # Main DataFrame: all users, but for multi-conv users, exclude their last conversation
    not_last_conv = ~is_last_conv | ~df['user_hashed_ip'].isin(multi_conv_users)
    main_df = df[not_last_conv]
    main_agg = main_df.groupby('user_hashed_ip').agg(
        conversation = ('user_content', lambda x: ' '.join(x)),
        num_turns = ('user_content', 'count'),
        num_conversations = ('conversation_hash', pd.Series.nunique)
    ).reset_index()
    for col in ['trait_openness', 'trait_consciousness', 'trait_extraversion', 'trait agreableness', 'trait_neuroticism']:
        main_agg[col] = None

    # Save to CSV
    main_csv = Path(__file__).parent.parent / 'data' / 'user_aggregate_conversations.csv'
    test_csv = Path(__file__).parent.parent / 'data' / 'user_aggregate_conversations_test.csv'
    main_agg.to_csv(main_csv, index=False)
    test_agg.to_csv(test_csv, index=False)
    print(f"Saved main DataFrame to: {main_csv}")
    print(f"Saved test DataFrame to: {test_csv}")
    print("\nSample of main DataFrame:")
    print(main_agg.head())
    print("\nSample of test DataFrame:")
    print(test_agg.head())
