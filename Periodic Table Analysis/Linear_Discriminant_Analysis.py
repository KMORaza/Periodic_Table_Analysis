import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
data = pd.read_csv('Modern_Periodic_Table.csv')
data = data.dropna(axis=1)
columns_to_keep = ['Atomic Number', 'Period', 'Electron Configuration']
data = data[columns_to_keep]
label_encoder = LabelEncoder()
data['Electron Configuration'] = label_encoder.fit_transform(data['Electron Configuration'])
X = data.drop(columns=['Period'])
y = data['Period']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Accuracy =", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
class_report_dict = classification_report(y_test, y_pred, output_dict=True)
report_tuples = [(key, value['precision'], value['recall'], value['f1-score'], value['support']) for key, value in class_report_dict.items() if key.isdigit()]
report_tuples.insert(0, ('Period', 'Precision', 'Recall', 'F1-score', 'Support'))
print(tabulate(report_tuples, headers='firstrow', tablefmt='fancy_grid'))
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=lda.classes_, yticklabels=lda.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
