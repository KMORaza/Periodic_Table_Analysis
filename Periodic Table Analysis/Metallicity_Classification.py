import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
df = pd.read_csv("Modern_Periodic_Table.csv")
df.dropna(subset=['Electrical Conductivity', 'Melting Point'], inplace=True)
features = ['Electrical Conductivity', 'Melting Point']
X = df[features]
df['Metallicity'] = pd.cut(df['Electrical Conductivity'], bins=[-float('inf'), 1000, float('inf')], labels=['Non-Metal', 'Metal'])
y = df['Metallicity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_imputed, y_train)
y_pred = classifier.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
