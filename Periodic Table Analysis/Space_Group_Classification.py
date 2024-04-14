import pandas as pd
from tabulate import tabulate
df = pd.read_csv("Modern_Periodic_Table.csv")
def classify_space_group(row):
    space_group_name = row['Space Group Name']
    space_group_number = row['Space Group Number']
    if pd.isna(space_group_name) and pd.isna(space_group_number):
        return 'Unknown'
    elif pd.isna(space_group_name):
        return f"Space Group {space_group_number}"
    else:
        return space_group_name
df['Space Group Classification'] = df.apply(classify_space_group, axis=1)
print(tabulate(df[['Element Name', 'Space Group Name', 'Space Group Number', 'Space Group Classification']], headers='keys', tablefmt='grid'))
