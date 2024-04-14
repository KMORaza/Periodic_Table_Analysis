import pandas as pd
from tabulate import tabulate
df = pd.read_csv("Modern_Periodic_Table.csv")
def classify_reactivity(row):
    electronegativity = row['Electronegativity']
    electron_affinity = row['Electron Affinity']
    autoignition_point = row['Autoignition Point']
    if electronegativity > 1.5 and electron_affinity > 0 and autoignition_point > 400:
        return 'Reactive'
    else:
        return 'Non-Reactive'
df['Reactivity'] = df.apply(classify_reactivity, axis=1)
print(tabulate(df[['Element Name', 'Electronegativity', 'Electron Affinity', 'Reactivity']], headers='keys', tablefmt='grid'))
