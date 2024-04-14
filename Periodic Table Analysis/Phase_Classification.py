import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report  # Importing classification_report
from sklearn.impute import SimpleImputer
df = pd.read_csv("Modern_Periodic_Table.csv")
df.dropna(subset=['Melting Point', 'Boiling Point', 'Atomic Number', 'Atomic Mass', 'Density'], inplace=True)
features = ['Atomic Number', 'Atomic Mass', 'Density', 'Melting Point', 'Boiling Point',
            'Heat of Fusion', 'Heat of Vaporization', 'Specific Heat']
X = df[features]
y = df['State']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_imputed, y_train)
y_pred = classifier.predict(X_test_imputed)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))  
