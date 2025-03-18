import pandas as pd

# Step 1: Load the ratings data
ratings_path = 'ml-1m/ratings.dat'  # Change to the correct path
columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Reading the file
ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=columns)

# Step 2: Sort movies by timestamp for each user
ratings_sorted = ratings.sort_values(by=['UserID', 'Timestamp'])

# Step 3: Group movies by UserID to create sequences
user_sequences = ratings_sorted.groupby('UserID')['MovieID'].apply(list).reset_index()

# Step 4: Save user sequences in a format for nanoGPT
with open('movielens_nanogpt_format.txt', 'w') as f:
    for _, row in user_sequences.iterrows():
        sequence = ' '.join(map(str, row['MovieID']))  # MovieID sequence as tokens
        f.write(f"{sequence}\n")  # Each user's sequence on a new line