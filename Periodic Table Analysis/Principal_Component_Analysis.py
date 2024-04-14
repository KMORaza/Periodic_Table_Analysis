import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
data = pd.read_csv('Modern_Periodic_Table.csv')
numeric_cols = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point', 'Absolute Melting Point',
                'Absolute Boiling Point', 'Critical Pressure', 'Critical Temperature', 'Heat of Fusion',
                'Heat of Vaporization', 'Heat of Combustion', 'Specific Heat', 'Adiabatic Index', 'Neel Point',
                'Thermal Conductivity', 'Thermal Expansion', 'Liquid Density', 'Molar Volume', 'Brinell Hardness',
                'Mohs Hardness', 'Vickers Hardness', 'Bulk Modulus', 'Shear Modulus', "Young's Modulus",
                "Poisson's Ratio", 'Refractive Index', 'Speed of Sound', 'Valency', 'Electronegativity',
                'Electron Affinity', 'Autoignition Point', 'Flash Point', 'Atomic Radius', 'Covalent Radius',
                'van der Waals Radius', 'Half-Life', 'Lifetime', 'Neutron Cross Section', 'Neutron Mass Absorption',
                'Graph Period', 'Graph Group']
numeric_data = data[numeric_cols]
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_data = numeric_data.apply(lambda x: np.where(x > 1e10, np.nan, x))
imputer = SimpleImputer(strategy='median')
imputed_data = imputer.fit_transform(numeric_data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed_data)
pca = PCA()
pca.fit(scaled_data)
principal_components = pca.components_
print("Explained Variance Ratio =", pca.explained_variance_ratio_)
print("\nPrincipal Components =", principal_components)
