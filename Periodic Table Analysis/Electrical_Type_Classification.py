import pandas as pd
from tabulate import tabulate
df = pd.read_csv("Modern_Periodic_Table.csv")
def classify_electrical_type(row):
    electrical_conductivity = row['Electrical Conductivity']
    resistivity = row['Resistivity']
    if electrical_conductivity > 1e6:
        return 'Conductor'
    elif electrical_conductivity <= 1e6 and resistivity <= 1e-4:
        return 'Semiconductor'
    else:
        return 'Insulator'
df['Electrical Type'] = df.apply(classify_electrical_type, axis=1)
print(tabulate(df[['Element Name', 'Electrical Conductivity', 'Resistivity', 'Electrical Type']], headers='keys', tablefmt='grid'))
