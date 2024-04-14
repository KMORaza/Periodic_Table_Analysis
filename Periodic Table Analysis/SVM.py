# Support Vecctor Machine (SVM)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
data = pd.read_csv("Periodic_Table.csv")
data.dropna(inplace=True)
X = data[['Atomic Mass', 'Boiling Point', 'Density', 'Melting Point', 'Molar Heat',
          'Atomic Number', 'Group', 'X Position', 'Y Position',
          'WX Position', 'WY Position', 'Electron Affinity']]
y = data['Period']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1],
              'kernel': ['linear', 'rbf', 'poly'],
              'class_weight': [None, 'balanced']}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
print("Best parameters:", grid_search.best_params_)
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy =", accuracy)
