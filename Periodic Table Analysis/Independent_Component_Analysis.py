import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
data = pd.read_csv("Modern_Periodic_Table.csv")
data.replace([np.inf, -np.inf], np.nan, inplace=True)
columns_without_nans = data.columns[data.notna().all()].tolist()
data_cleaned = data[columns_without_nans].dropna()
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned[numeric_columns])
ica = FastICA(n_components=len(numeric_columns), random_state=42)
ica_result = ica.fit_transform(data_scaled)
ica_df = pd.DataFrame(ica_result, columns=[f'ICA-{i}' for i in range(1, len(numeric_columns)+1)])
result_df = pd.concat([data_cleaned.reset_index(drop=True), ica_df], axis=1)
print(tabulate(result_df, headers='keys', tablefmt='psql'))
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result_df['ICA-1'], result_df['ICA-2'], result_df['ICA-3'], c='b', marker='o', alpha=0.5)
ax.set_title('Independent Component Analysis (ICA)')
ax.set_xlabel('ICA-1')
ax.set_ylabel('ICA-2')
ax.set_zlabel('ICA-3')
plt.show()
mixing_matrix = ica.mixing_
independent_components = ica.components_
component_kurtosis = kurtosis(independent_components, axis=1)
def negentropy(X):
    def g(x):
        return np.tanh(x / 2)
    return np.mean(np.log(np.cosh(X)) - np.log(np.sqrt(np.mean(g(X)**2)) + 1e-10))
component_negentropy = [negentropy(ic) for ic in independent_components]
ica_report = f"""
\n\n📋 Independent Component Analysis (ICA) Report:
------------------------------------------------
Number of components = {len(independent_components)}
\nMixing Matrix:
{mixing_matrix}
\nIndependent Components:
{independent_components}
\nKurtosis of Independent Components = {component_kurtosis}
\nNegentropy of Independent Components = {component_negentropy}
"""
print(ica_report)
