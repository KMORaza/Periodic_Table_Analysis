import pandas as pd
from tabulate import tabulate
df = pd.read_csv("Modern_Periodic_Table.csv")
def classify_lifetime(row):
    decay_mode = row['Decay Mode']
    half_life = row['Half-Life']
    if pd.isna(decay_mode) and pd.isna(half_life):
        return 'Unknown'
    elif pd.isna(half_life):
        return f"Decay mode: {decay_mode}"
    elif pd.isna(decay_mode):
        return f"Half life: {half_life} s"
    else:
        return f"Half-life: {half_life} s"
df['Lifetime Classification'] = df.apply(classify_lifetime, axis=1)
print(tabulate(df[['Element Name', 'Half-Life', 'Lifetime Classification']], headers='keys', tablefmt='grid'))
