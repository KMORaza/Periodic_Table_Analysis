import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
df = pd.read_csv("Modern_Periodic_Table.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric = df.select_dtypes(include=['number'])
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)
n_factors = 10
fa = FactorAnalyzer(n_factors, rotation=None)
fa.fit(df_scaled)
factor_loadings = pd.DataFrame(fa.loadings_, index=df_numeric.columns, columns=[f'Factor {i+1}' for i in range(n_factors)])
print("Factor Loadings:")
print(tabulate(factor_loadings, headers='keys', tablefmt='psql'))
explained_variance = fa.get_factor_variance()
explained_variance_df = pd.DataFrame(explained_variance, index=['Sum of Squared Loadings', 'Proportion Var', 'Cumulative Var'], columns=[f'Factor {i+1}' for i in range(n_factors)])
print("\nExplained Variance:")
print(tabulate(explained_variance_df, headers='keys', tablefmt='psql'))
eigenvalues = fa.get_eigenvalues()
print("\nEigenvalues:")
print(eigenvalues)
for i, eigenvalue in enumerate(eigenvalues, start=1):
    print(f"\nFactor {i} = {eigenvalue}")
communalities = fa.get_communalities()
print("\nCommunalities:")
print(tabulate(pd.DataFrame(communalities, index=df_numeric.columns, columns=["Communalities"]), headers='keys', tablefmt='psql'))
total_variance = np.sum(fa.get_factor_variance()[0])
print("\nTotal Variance Explained by Factors =", total_variance)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()
