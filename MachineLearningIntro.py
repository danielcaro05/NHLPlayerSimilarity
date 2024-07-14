import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

conn = sqlite3.connect('NSTdata.db')
c = conn.cursor()

data = pd.read_sql_query("SELECT * from 'Skater_Data' WHERE TOI_All > 100", conn)

def normalize_position(position):
    if position == 'D':
        return 'D'  # Defenseman
    return 'F'  # Forward (or any other non-defenseman)

data['Normalized_Position'] = data['Position'].apply(normalize_position)

weights = {'Goals_All': 0, 'First Assists_All': 0, 'Second Assists_All': 0, 'Total Assists_All': 0, 'Total Points_All': 0,
           'Goals_5v5': 0, 'First Assists_5v5': 0, 'Second Assists_5v5': 0, 'Total Assists_5v5': 0, 'Total Points_5v5': 0,
           'Goals_PP': 0, 'First Assists_PP': 0, 'Second Assists_PP': 0, 'Total Assists_PP': 0, 'Total Points_PP': 0,
           'Goals_PK': 0, 'First Assists_PK': 0, 'Second Assists_PK': 0, 'Total Assists_PK': 0, 'Total Points_PK': 0,

           'TOI_All': 0, 'Giveaways_All': 0, 'Takeaways_All': 0, 'Hits_All': 0, 'Hits Taken_All': 0, 'Shots Blocked_All': 0,
           'TOI_5v5': 0, 'Giveaways_5v5': 0, 'Takeaways_5v5': 0, 'Hits_5v5': 0, 'Hits Taken_5v5': 0, 'Shots Blocked_5v5': 0,
           'TOI_PP': 0, 'Giveaways_PP': 0, 'Takeaways_PP': 0, 'Hits_PP': 0, 'Hits Taken_PP': 0, 'Shots Blocked_PP': 0,
           'TOI_PK': 0, 'Giveaways_PK': 0, 'Takeaways_PK': 0, 'Hits_PK': 0, 'Hits Taken_PK': 0, 'Shots Blocked_PK': 0,

           'PIM_All': 0, 'Total Penalties_All': 0, 'Minor_All': 0, 'Major_All': 0, 'Misconduct_All': 0, 'Penalties Drawn_All': 0,
           'PIM_5v5': 0, 'Total Penalties_5v5': 0, 'Minor_5v5': 0, 'Major_5v5': 0, 'Misconduct_5v5': 0, 'Penalties Drawn_5v5': 0,
           'PIM_PP': 0, 'Total Penalties_PP': 0, 'Minor_PP': 0, 'Major_PP': 0, 'Misconduct_PP': 0, 'Penalties Drawn_PP': 0,
           'PIM_PK': 0, 'Total Penalties_PK': 0, 'Minor_PK': 0, 'Major_PK': 0, 'Misconduct_PK': 0, 'Penalties Drawn_PK': 0,

           'Shots_All': 0, 'ixG_All': 0, 'iCF_All': 0, 'iFF_All': 0, 'iSCF_All': 0, 'iHDCF_All': 0,
           'Shots_5v5': 0, 'ixG_5v5': 0, 'iCF_5v5': 0, 'iFF_5v5': 0, 'iSCF_5v5': 0, 'iHDCF_5v5': 0,
           'Shots_PP': 0, 'ixG_PP': 0, 'iCF_PP': 0, 'iFF_PP': 0, 'iSCF_PP': 0, 'iHDCF_PP': 0,
           'Shots_PK': 0, 'ixG_PK': 0, 'iCF_PK': 0, 'iFF_PK': 0, 'iSCF_PK': 0, 'iHDCF_PK': 0,

           'Rush Attempts_All': 0, 'Rebounds Created_All': 0, 'Faceoffs Won_All': 0, 'Faceoffs Lost_All': 0,
           'Rush Attempts_5v5': 0, 'Rebounds Created_5v5': 0, 'Faceoffs Won_5v5': 0, 'Faceoffs Lost_5v5': 0,
           'Rush Attempts_PP': 0, 'Rebounds Created_PP': 0, 'Faceoffs Won_PP': 0, 'Faceoffs Lost_PP': 0,
           'Rush Attempts_PK': 0, 'Rebounds Created_PK': 0, 'Faceoffs Won_PK': 0, 'Faceoffs Lost_PK': 0,

           'CF_All': 0, 'CA_All': 0, 'FF_All': 0, 'FA_All': 0, 'SF_All': 0, 'SA_All': 0, 'GF_All': 0, 'GA_All': 0,
           'CF_5v5': 0, 'CA_5v5': 0, 'FF_5v5': 0, 'FA_5v5': 0, 'SF_5v5': 0, 'SA_5v5': 0, 'GF_5v5': 0, 'GA_5v5': 0,
           'CF_PP': 0, 'CA_PP': 0, 'FF_PP': 0, 'FA_PP': 0, 'SF_PP': 0, 'SA_PP': 0, 'GF_PP': 0, 'GA_PP': 0,
           'CF_PK': 0, 'CA_PK': 0, 'FF_PK': 0, 'FA_PK': 0, 'SF_PK': 0, 'SA_PK': 0, 'GF_PK': 0, 'GA_PK': 0,

           'xGF%_All': 0, 'CF%_All': 0, 'FF%_All': 0, 'SF%_All': 0, 'GF%_All': 0, 'SCF%_All': 0, 'HDCF%_All': 0, 'HDGF%_All': 0,
           'xGF%_5v5': 1, 'CF%_5v5': 0, 'FF%_5v5': 0, 'SF%_5v5': 0, 'GF%_5v5': 0, 'SCF%_5v5': 0, 'HDCF%_5v5': 0, 'HDGF%_5v5': 0,

           'MDCF%_All': 0, 'MDGF%_All': 0, 'LDCF%_All': 0, 'LDGF%_All': 0, 'On-Ice SH%_All': 0, 'On-Ice SV%_All': 0,
           'MDCF%_5v5': 0, 'MDGF%_5v5': 0, 'LDCF%_5v5': 0, 'LDGF%_5v5': 0, 'On-Ice SH%_5v5': 0, 'On-Ice SV%_5v5': 0,

           'xGF_All': 0, 'xGA_All': 0, 'SCF_All': 0, 'SCA_All': 0,
           'xGF_5v5': 0, 'xGA_5v5': 0, 'SCF_5v5': 0, 'SCA_5v5': 0,
           'xGF_PP': 0, 'xGA_PP': 0, 'SCF_PP': 0, 'SCA_PP': 0,
           'xGF_PK': 0, 'xGA_PK': 0, 'SCF_PK': 0, 'SCA_PK': 0,

           'HDCF_All': 0, 'HDCA_All': 0, 'HDGF_All': 0, 'HDGA_All': 0, 'MDCF_All': 0, 'MDCA_All': 0,
           'HDCF_5v5': 0, 'HDCA_5v5': 0, 'HDGF_5v5': 0, 'HDGA_5v5': 0, 'MDCF_5v5': 0, 'MDCA_5v5': 0,
           'HDCF_PP': 0, 'HDCA_PP': 0, 'HDGF_PP': 0, 'HDGA_PP': 0, 'MDCF_PP': 0, 'MDCA_PP': 0,
           'HDCF_PK': 0, 'HDCA_PK': 0, 'HDGF_PK': 0, 'HDGA_PK': 0, 'MDCF_PK': 0, 'MDCA_PK': 0,

           'MDGF_All': 0, 'MDGA_All': 0, 'LDCF_All': 0, 'LDCA_All': 0, 'LDGF_All': 0, 'LDGA_All': 0,
           'MDGF_5v5': 0, 'MDGA_5v5': 0, 'LDCF_5v5': 0, 'LDCA_5v5': 0, 'LDGF_5v5': 0, 'LDGA_5v5': 0,
           'MDGF_PP': 0, 'MDGA_PP': 0, 'LDCF_PP': 0, 'LDCA_PP': 0, 'LDGF_PP': 0, 'LDGA_PP': 0,
           'MDGF_PK': 0, 'MDGA_PK': 0, 'LDCF_PK': 0, 'LDCA_PK': 0, 'LDGF_PK': 0, 'LDGA_PK': 0,

           'Off. Zone Starts_All': 0, 'Neu. Zone Starts_All': 0, 'Def. Zone Starts_All': 0, 'On The Fly Stats_All': 0,
           'Off. Zone Starts_5v5': 0, 'Neu. Zone Starts_5v5': 0, 'Def. Zone Starts_5v5': 0, 'On The Fly Stats_5v5': 0,
           'Off. Zone Starts_PP': 0, 'Neu. Zone Starts_PP': 0, 'Def. Zone Starts_PP': 0, 'On The Fly Stats_PP': 0,
           'Off. Zone Starts_PK': 0, 'Neu. Zone Starts_PK': 0, 'Def. Zone Starts_PK': 0, 'On The Fly Stats_PK': 0,

           'Off. Zone Start %_All': 0, 'Off. Zone Faceoffs_All': 0, 'Neu. Zone Faceoffs_All': 0, 'Def. Zone Faceoffs_All': 0,
           'Off. Zone Start %_5v5': 1, 'Off. Zone Faceoffs_5v5': 0, 'Neu. Zone Faceoffs_5v5': 0, 'Def. Zone Faceoffs_5v5': 0,
           'Neu. Zone Faceoffs_PP': 0, 'Def. Zone Faceoffs_PP': 0,
           'Neu. Zone Faceoffs_PK': 0, 'Def. Zone Faceoffs_PK': 0,



           # I want this extended basically just a repeat of everything i just wrote except add '/GP' to each one
           'Goals_All/GP': 1, 'First Assists_All/GP': 1, 'Second Assists_All/GP': 0, 'Total Assists_All/GP': 0, 'Total Points_All/GP': 0,
           'Goals_5v5/GP': 0, 'First Assists_5v5/GP': 0, 'Second Assists_5v5/GP': 0, 'Total Assists_5v5/GP': 0, 'Total Points_5v5/GP': 0,
           'Goals_PP/GP': 0, 'First Assists_PP/GP': 0, 'Second Assists_PP/GP': 0, 'Total Assists_PP/GP': 0, 'Total Points_PP/GP': 0,
           'Goals_PK/GP': 0, 'First Assists_PK/GP': 0, 'Second Assists_PK/GP': 0, 'Total Assists_PK/GP': 0, 'Total Points_PK/GP': 0,

           'TOI_All/GP': 0, 'Giveaways_All/GP': 0, 'Takeaways_All/GP': 0, 'Hits_All/GP': 0, 'Hits Taken_All/GP': 0, 'Shots Blocked_All/GP': 0,
           'TOI_5v5/GP': 1, 'Giveaways_5v5/GP': 0, 'Takeaways_5v5/GP': 0, 'Hits_5v5/GP': 0, 'Hits Taken_5v5/GP': 0, 'Shots Blocked_5v5/GP': 0,
           'TOI_PP/GP': 0.5, 'Giveaways_PP/GP': 0, 'Takeaways_PP/GP': 0, 'Hits_PP/GP': 0, 'Hits Taken_PP/GP': 0, 'Shots Blocked_PP/GP': 0,
           'TOI_PK/GP': 0.5, 'Giveaways_PK/GP': 0, 'Takeaways_PK/GP': 0, 'Hits_PK/GP': 0, 'Hits Taken_PK/GP': 0, 'Shots Blocked_PK/GP': 0,

           'PIM_All/GP': 0, 'Total Penalties_All/GP': 0, 'Minor_All/GP': 0, 'Major_All/GP': 0, 'Misconduct_All/GP': 0, 'Penalties Drawn_All/GP': 0,
           'PIM_5v5/GP': 0, 'Total Penalties_5v5/GP': 0, 'Minor_5v5/GP': 0, 'Major_5v5/GP': 0, 'Misconduct_5v5/GP': 0, 'Penalties Drawn_5v5/GP': 0,
           'PIM_PP/GP': 0, 'Total Penalties_PP/GP': 0, 'Minor_PP/GP': 0, 'Major_PP/GP': 0, 'Misconduct_PP/GP': 0, 'Penalties Drawn_PP/GP': 0,
           'PIM_PK/GP': 0, 'Total Penalties_PK/GP': 0, 'Minor_PK/GP': 0, 'Major_PK/GP': 0, 'Misconduct_PK/GP': 0, 'Penalties Drawn_PK/GP': 0,

           'Shots_All/GP': 0, 'ixG_All/GP': 0, 'iCF_All/GP': 0, 'iFF_All/GP': 0, 'iSCF_All/GP': 0, 'iHDCF_All/GP': 0,
           'Shots_5v5/GP': 0, 'ixG_5v5/GP': 0, 'iCF_5v5/GP': 0, 'iFF_5v5/GP': 0, 'iSCF_5v5/GP': 0, 'iHDCF_5v5/GP': 0,
           'Shots_PP/GP': 0, 'ixG_PP/GP': 0, 'iCF_PP/GP': 0, 'iFF_PP/GP': 0, 'iSCF_PP/GP': 0, 'iHDCF_PP/GP': 0,
           'Shots_PK/GP': 0, 'ixG_PK/GP': 0, 'iCF_PK/GP': 0, 'iFF_PK/GP': 0, 'iSCF_PK/GP': 0, 'iHDCF_PK/GP': 0,

           'Rush Attempts_All/GP': 0, 'Rebounds Created_All/GP': 0, 'Faceoffs Won_All/GP': 0, 'Faceoffs Lost_All/GP': 0,
           'Rush Attempts_5v5/GP': 0, 'Rebounds Created_5v5/GP': 0, 'Faceoffs Won_5v5/GP': 0, 'Faceoffs Lost_5v5/GP': 0,
           'Rush Attempts_PP/GP': 0, 'Rebounds Created_PP/GP': 0, 'Faceoffs Won_PP/GP': 0, 'Faceoffs Lost_PP/GP': 0,
           'Rush Attempts_PK/GP': 0, 'Rebounds Created_PK/GP': 0, 'Faceoffs Won_PK/GP': 0, 'Faceoffs Lost_PK/GP': 0,

           'CF_All/GP': 0, 'CA_All/GP': 0, 'FF_All/GP': 0, 'FA_All/GP': 0, 'SF_All/GP': 0, 'SA_All/GP': 0, 'GF_All/GP': 0, 'GA_All/GP': 0,
           'CF_5v5/GP': 0, 'CA_5v5/GP': 0, 'FF_5v5/GP': 0, 'FA_5v5/GP': 0, 'SF_5v5/GP': 0, 'SA_5v5/GP': 0, 'GF_5v5/GP': 0, 'GA_5v5/GP': 0,
           'CF_PP/GP': 0, 'CA_PP/GP': 0, 'FF_PP/GP': 0, 'FA_PP/GP': 0, 'SF_PP/GP': 0, 'SA_PP/GP': 0, 'GF_PP/GP': 0, 'GA_PP/GP': 0,
           'CF_PK/GP': 0, 'CA_PK/GP': 0, 'FF_PK/GP': 0, 'FA_PK/GP': 0, 'SF_PK/GP': 0, 'SA_PK/GP': 0, 'GF_PK/GP': 0, 'GA_PK/GP': 0,

           'CF%_All/GP': 0, 'FF%_All/GP': 0, 'SF%_All/GP': 0, 'GF%_All/GP': 0, 'SCF%_All/GP': 0, 'HDCF%_All/GP': 0, 'HDGF%_All/GP': 0,
           'CF%_5v5/GP': 0, 'FF%_5v5/GP': 0, 'SF%_5v5/GP': 0, 'GF%_5v5/GP': 0, 'SCF%_5v5/GP': 0, 'HDCF%_5v5/GP': 0, 'HDGF%_5v5/GP': 0,

           'MDCF%_All/GP': 0, 'MDGF%_All/GP': 0, 'LDCF%_All/GP': 0, 'LDGF%_All/GP': 0, 'On-Ice SH%_All/GP': 0, 'On-Ice SV%_All/GP': 0,
           'MDCF%_5v5/GP': 0, 'MDGF%_5v5/GP': 0, 'LDCF%_5v5/GP': 0, 'LDGF%_5v5/GP': 0, 'On-Ice SH%_5v5/GP': 0, 'On-Ice SV%_5v5/GP': 0,

           'xGF_All/GP': 0, 'xGA_All/GP': 0, 'SCF_All/GP': 0, 'SCA_All/GP': 0,
           'xGF_5v5/GP': 0, 'xGA_5v5/GP': 0, 'SCF_5v5/GP': 0, 'SCA_5v5/GP': 0,
           'xGF_PP/GP': 0, 'xGA_PP/GP': 0, 'SCF_PP/GP': 0, 'SCA_PP/GP': 0,
           'xGF_PK/GP': 0, 'xGA_PK/GP': 0, 'SCF_PK/GP': 0, 'SCA_PK/GP': 0,

           'HDCF_All/GP': 0, 'HDCA_All/GP': 0, 'HDGF_All/GP': 0, 'HDGA_All/GP': 0, 'MDCF_All/GP': 0, 'MDCA_All/GP': 0,
           'HDCF_5v5/GP': 0, 'HDCA_5v5/GP': 0, 'HDGF_5v5/GP': 0, 'HDGA_5v5/GP': 0, 'MDCF_5v5/GP': 0, 'MDCA_5v5/GP': 0,
           'HDCF_PP/GP': 0, 'HDCA_PP/GP': 0, 'HDGF_PP/GP': 0, 'HDGA_PP/GP': 0, 'MDCF_PP/GP': 0, 'MDCA_PP/GP': 0,
           'HDCF_PK/GP': 0, 'HDCA_PK/GP': 0, 'HDGF_PK/GP': 0, 'HDGA_PK/GP': 0, 'MDCF_PK/GP': 0, 'MDCA_PK/GP': 0,

           'MDGF_All/GP': 0, 'MDGA_All/GP': 0, 'LDCF_All/GP': 0, 'LDCA_All/GP': 0, 'LDGF_All/GP': 0, 'LDGA_All/GP': 0,
           'MDGF_5v5/GP': 0, 'MDGA_5v5/GP': 0, 'LDCF_5v5/GP': 0, 'LDCA_5v5/GP': 0, 'LDGF_5v5/GP': 0, 'LDGA_5v5/GP': 0,
           'MDGF_PP/GP': 0, 'MDGA_PP/GP': 0, 'LDCF_PP/GP': 0, 'LDCA_PP/GP': 0, 'LDGF_PP/GP': 0, 'LDGA_PP/GP': 0,
           'MDGF_PK/GP': 0, 'MDGA_PK/GP': 0, 'LDCF_PK/GP': 0, 'LDCA_PK/GP': 0, 'LDGF_PK/GP': 0, 'LDGA_PK/GP': 0,

           'Off. Zone Starts_All/GP': 0, 'Neu. Zone Starts_All/GP': 0, 'Def. Zone Starts_All/GP': 0, 'On The Fly Stats_All/GP': 0,
           'Off. Zone Starts_5v5/GP': 0, 'Neu. Zone Starts_5v5/GP': 0, 'Def. Zone Starts_5v5/GP': 0, 'On The Fly Stats_5v5/GP': 0,
           'Off. Zone Starts_PP/GP': 0, 'Neu. Zone Starts_PP/GP': 0, 'Def. Zone Starts_PP/GP': 0, 'On The Fly Stats_PP/GP': 0,
           'Off. Zone Starts_PK/GP': 0, 'Neu. Zone Starts_PK/GP': 0, 'Def. Zone Starts_PK/GP': 0, 'On The Fly Stats_PK/GP': 0,

           'Off. Zone Start %_All/GP': 0, 'Off. Zone Faceoffs_All/GP': 0, 'Neu. Zone Faceoffs_All/GP': 0, 'Def. Zone Faceoffs_All/GP': 0,
           'Off. Zone Start %_5v5/GP': 0, 'Off. Zone Faceoffs_5v5/GP': 0, 'Neu. Zone Faceoffs_5v5/GP': 0, 'Def. Zone Faceoffs_5v5/GP': 0,
           'Neu. Zone Faceoffs_PP/GP': 0, 'Def. Zone Faceoffs_PP/GP': 0,
           'Neu. Zone Faceoffs_PK/GP': 0, 'Def. Zone Faceoffs_PK/GP': 0
           }

features = []
for key in weights:
    if weights[key] != 0:
        features.append(key)

player_data = data[features]
players = data['Player']
positions = data['Normalized_Position']  # Use normalized positions

scaler = StandardScaler()
player_data = scaler.fit_transform(player_data)

weighted_player_data = player_data * [weights[feature] for feature in features]

normalized_player_data = pd.DataFrame(weighted_player_data, columns=features)

# Merge the player names with the normalized feature data
merged_data = pd.concat([players.reset_index(drop=True), positions.reset_index(drop=True), normalized_player_data], axis=1)

distance_matrix = euclidean_distances(normalized_player_data)

# Create a DataFrame for the similarity matrix
distance_df = pd.DataFrame(distance_matrix, index=merged_data['Player'], columns=merged_data['Player'])


def find_similar_players(player_name, similarity_df, top_n=5):
    if player_name not in distance_df.index:
        return f"Player {player_name} not found in the dataset."

        # Get the normalized position of the player
    player_position = merged_data.loc[merged_data['Player'] == player_name, 'Normalized_Position'].values[0]

    # Filter the players by the same normalized position
    if player_position == 'D':
        same_position_players = merged_data[merged_data['Normalized_Position'] == 'D']['Player']
    else:
        same_position_players = merged_data[merged_data['Normalized_Position'] != 'D']['Player']

    filtered_distance_df = distance_df.loc[same_position_players, same_position_players]

    # Get the distance scores for the given player
    distance_scores = filtered_distance_df[player_name]

    # Sort the scores in ascending order (because lower distance means more similarity) and exclude the player itself
    similar_players = distance_scores.sort_values(ascending=True).drop(player_name).head(top_n)

    return similar_players


# Example usage
player_name = 'Auston Matthews'  # Replace with a player name from your dataset
top_n = 10
similar_players = find_similar_players(player_name, distance_df, top_n)
print(similar_players)

conn.close()
