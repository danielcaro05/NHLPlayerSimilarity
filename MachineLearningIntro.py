import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

conn = sqlite3.connect('NSTdata.db')

data = pd.read_sql_query("SELECT * from 'All/5v5_Skaters' WHERE TOI_Individual_All > 500", conn)

features = ['Goals_All', 'Total Assists_All']
player_data = data[features]
players = data['Player']

scaler = StandardScaler()
player_data = scaler.fit_transform(player_data)

normalized_player_data = pd.DataFrame(player_data, columns=features)

# Merge the player names with the normalized feature data
merged_data = pd.concat([players.reset_index(drop=True), normalized_player_data], axis=1)

distance_matrix = euclidean_distances(normalized_player_data)

# Create a DataFrame for the similarity matrix
distance_df = pd.DataFrame(distance_matrix, index=merged_data['Player'], columns=merged_data['Player'])


def find_similar_players(player_name, similarity_df, top_n=5):
    if player_name not in similarity_df.index:
        return f"Player {player_name} not found in the dataset."

    # Get the distance scores for the given player
    distance_scores = distance_df[player_name]

    # Sort the scores in ascending order (because lower distance means more similarity) and exclude the player itself
    similar_players = distance_scores.sort_values(ascending=True).drop(player_name).head(top_n)

    return similar_players


# Example usage
player_name = 'Blake Coleman'  # Replace with a player name from your dataset
top_n = 5
similar_players = find_similar_players(player_name, distance_df, top_n)
print(similar_players)

conn.close()
