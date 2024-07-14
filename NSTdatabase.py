import sqlite3
import os
import pandas as pd


# Connect to the SQLite database
conn = sqlite3.connect('NSTdata.db')

df = pd.read_sql_query("SELECT * from 'Skater_Data'", conn)
real_df = pd.read_sql_query("SELECT * from 'Skater_Data'", conn)

columns = real_df.columns
columns_with_spaces = [col for col in columns if '\xa0' in col]

for col in columns_with_spaces:
    print(col)
# real_df.to_sql('Skater_Data', conn, if_exists='replace')

# Adam Gaudette

# Close the connection to the SQLite database
conn.close()

# Below is the code that I used to merge all of the tables in the database into one table

# df_5v5_Skaters = pd.read_sql_query("SELECT * from 'All/5v5_Skaters'", conn)
# df_All_Skaters = pd.read_sql_query("SELECT * from 'PP/PK_Skaters'", conn)
#
# # Merge the two DataFrames on the 'Player' column
# merged_df = pd.merge(df_5v5_Skaters, df_All_Skaters, on=['Player', 'Team', 'Position', 'GP'], how='outer', suffixes=('_PP', '_PK'))
# merged_df = merged_df.drop(columns=['index_PP', 'index_PK'])
# merged_df = merged_df.fillna('-')
#
#
