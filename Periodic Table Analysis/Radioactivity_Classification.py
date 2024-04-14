import pandas as pd
from tabulate import tabulate
df = pd.read_csv("Modern_Periodic_Table.csv")
def classify_radioactivity(row):
    half_life = row['Half-Life']
    decay_mode = row['Decay Mode']
    neutron_cross_section = row['Neutron Cross Section']
    if pd.isna(half_life) and pd.isna(decay_mode) and pd.isna(neutron_cross_section):
        return 'Non-Radioactive'
    else:
        return 'Radioactive'
df['Radioactivity'] = df.apply(classify_radioactivity, axis=1)
print(tabulate(df[['Element Name', 'Half-Life', 'Neutron Cross Section', 'Radioactivity']], headers='keys', tablefmt='grid'))
