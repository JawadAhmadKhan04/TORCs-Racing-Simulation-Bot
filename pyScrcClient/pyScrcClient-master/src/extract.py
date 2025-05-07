import pandas as pd

# Load your CSV file
df = pd.read_csv('merged_output.csv')


# Identify opponent and track columns
opponent_cols = [col for col in df.columns if col.startswith('opponent_')]
track_cols = [col for col in df.columns if col.startswith('track_')]

# Get all other columns
other_cols = [col for col in df.columns if col not in opponent_cols + track_cols]

# Construct new order: other + opponent + track
new_column_order = other_cols + opponent_cols + track_cols

# Reorder and save
df = df[new_column_order]
df.to_csv('reordered_output.csv', index=False)

print(f"✅ Reordered CSV saved as 'reordered_output.csv'")
