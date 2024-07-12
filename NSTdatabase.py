import sqlite3
import os
import pandas as pd

def has_duplicates(lst):
    return len(lst) != len(set(lst))



# Connect to the SQLite database
conn = sqlite3.connect('NSTdata.db')


df_5v5_Skaters = pd.read_sql_query("SELECT * from '5v5_Skaters'", conn)
df_All_Skaters = pd.read_sql_query("SELECT * from 'All_Skaters'", conn)

# Merge the two DataFrames on the 'Player' column
merged_df = pd.merge(df_5v5_Skaters, df_All_Skaters, on=['Player', 'Team'], suffixes=('_5v5', '_All'))

merged_df.to_sql('All/5v5_Skaters', conn, if_exists='replace')


# Close the connection to the SQLite database
conn.close()