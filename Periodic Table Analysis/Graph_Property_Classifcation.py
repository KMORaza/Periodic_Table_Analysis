import pandas as pd
from tabulate import tabulate
df = pd.read_csv("Modern_Periodic_Table.csv")
def classify_graph_properties(row):
    graph_period = row['Graph Period']
    graph_group = row['Graph Group']
    if pd.isna(graph_period) and pd.isna(graph_group):
        return 'Unknown'
    elif pd.isna(graph_period):
        return f"Group {graph_group}"
    elif pd.isna(graph_group):
        return f"Period {graph_period}"
    else:
        return f"Period {graph_period}, Group {graph_group}"
df['Graph Properties Classification'] = df.apply(classify_graph_properties, axis=1)
print(tabulate(df[['Element Name', 'Graph Period', 'Graph Group', 'Graph Properties Classification']], headers='keys', tablefmt='grid'))
