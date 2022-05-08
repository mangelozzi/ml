import pandas as pd

# LOAD DATA
# ---------
## CSV
df = pd.read_csv('pokemon_data.csv')
## Tab separation
# df_tabs = pd.read_csv('pokemon_data.xlsx', delimiter="\t"
## Excel file
# df_excel = pd.read_excel('pokemon_data.xlsx', delimiter="\t"

# PRINT
# -----
# print(df)
# print(df.head(5))
# print(df.tail(3))
# print(df.columns)
# print(df.Attack)  # Print a single column of heading 'Attack'
# print(df['Attack'])  # If has spaces etc, preferred method
# print(df['Attack'][0:5])

# nested brackets because attribute look up a value, but we passing in a list
# print(df[['Attack', 'Name', 'Type 1']])  # I.e. A list of colums
# print(df.iloc[30])  # slice 30'th row, iloc = integer location, the first item is index 0
# print(df.iloc[0:31])  # iloc is for [:] slicing
# print(df.iloc[30,1])  # Access single cell, 'ROW,COLUMN' notation

# ITERATE
# -------
# To iterate through enumerated rows:
# for i, row in df.iterrows():
#     print(i, row, row['Name'])

# SEARCH
# ------
# Search for rows matching a criteria:
# print(df.loc[df['Type 1'] == "Electric"])

# DESCRIBE
# --------
# print(df.describe())  # print count/mean/min/max/std deviation

# SORT
#-----
# print(df.sort_values('Name'))
# print(df.sort_values('Name', ascending=False))  # Reverse order
# print(df.sort_values(['Type 1', 'Name'], ascending=[True, False]))  # Sort by Type1 then Name

# EDIT DATA
# ---------

# Slow and easy
# df['New column'] = df['HP'] + df['Attack'] + df['Defense'] # Create a column based on the others

# 1. First we slice the columns/rows of interest
# 2. Then we sum the rows. 'axis=1' means sum horizontally, 'axis=0' means vertically
# df['New column'] = df.iloc[:, 4:10].sum(axis=1)  # [:, = all the rows 4:10] = 4th to 9th column (start of 10th)

# Rearranging columns
# cols = list(df.columns)
# df = df[cols[0:2] + [cols[-1]] + cols[2:-1]]

# print(list(df.columns))
# print(list(df.columns.values))
# df = df.drop(columns=['Name', 'HP'])  # Delete columns, WARNING, does NOT modify in place

# WRITE
# df.to_csv('pokemon_modified.csv')
# df.to_csv('pokemon_modified.csv', index=False)  # don't save the added index made by pandas
print(df.groupby(['Type 1']).count())

# print(df)
