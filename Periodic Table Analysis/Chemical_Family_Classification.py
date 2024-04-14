import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style
df = pd.read_csv("Modern_Periodic_Table.csv")
def classify_chemical_family(row):
    group = row['Group']
    period = row['Period']
    atomic_number = row['Atomic Number']
    if group == 1:
        return Fore.BLUE + 'Alkali Metals' + Style.RESET_ALL
    elif group == 2:
        return Fore.GREEN + 'Alkaline Earth Metals' + Style.RESET_ALL
    elif group == 17:
        return Fore.RED + 'Halogens' + Style.RESET_ALL
    elif group == 18:
        return Fore.MAGENTA + 'Noble Gases' + Style.RESET_ALL
    elif group >= 3 and group <= 12:
        return Fore.YELLOW + 'Transition Metals' + Style.RESET_ALL
    elif group >= 13 and group <= 16:
        if period == 2:
            return Fore.CYAN + 'Other Metals' + Style.RESET_ALL
        elif atomic_number in [5, 14, 32, 33, 51, 52, 84, 85]:
            return Fore.CYAN + 'Metalloids' + Style.RESET_ALL
        else:
            return Fore.CYAN + 'Other Non-Metals' + Style.RESET_ALL
    else:
        return 'Unknown'
df['Chemical Family'] = df.apply(classify_chemical_family, axis=1)
print(tabulate(df[['Element Name', 'Group', 'Period', 'Chemical Family']], headers='keys', tablefmt='grid'))
