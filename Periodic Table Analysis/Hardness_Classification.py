import pandas as pd
from tabulate import tabulate
df = pd.read_csv("Modern_Periodic_Table.csv")
def classify_hardness(row):
    brinell_hardness = row['Brinell Hardness']
    mohs_hardness = row['Mohs Hardness']
    vickers_hardness = row['Vickers Hardness']
    if brinell_hardness >= 400:
        return 'Very Hard'
    elif brinell_hardness >= 200 and brinell_hardness < 400:
        return 'Hard'
    elif mohs_hardness >= 6:
        return 'Hard'
    elif vickers_hardness >= 600:
        return 'Hard'
    elif brinell_hardness >= 100 and brinell_hardness < 200:
        return 'Medium Hard'
    else:
        return 'Soft'
df['Hardness Category'] = df.apply(classify_hardness, axis=1)
print(tabulate(df[['Element Name', 'Brinell Hardness', 'Mohs Hardness', 'Vickers Hardness', 'Hardness Category']], headers='keys', tablefmt='grid'))
