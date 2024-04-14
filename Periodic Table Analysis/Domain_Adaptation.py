import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
class DomainAdaptation:
    def __init__(self, data):
        self.data = data
    def train_test_split(self, test_size=0.2):
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size)
    def preprocess_data(self):
        self.train_data = self.train_data.dropna()
        self.test_data = self.test_data.dropna()
    def fit_source_model(self):
        self.model = LinearRegression()
        self.model.fit(self.train_data.drop(columns=['Boiling Point']), self.train_data['Boiling Point'])
        print("Coefficients of the fitted linear regression model:")
        for feature, coef in zip(self.train_data.drop(columns=['Boiling Point']).columns, self.model.coef_):
            print(f"{coef} ({feature})")
    def evaluate_on_target(self, target_data):
        target_data = target_data.dropna()
        target_X = target_data.drop(columns=['Boiling Point'])
        target_y_true = target_data['Boiling Point']
        target_y_pred = self.model.predict(target_X)
        mse = mean_squared_error(target_y_true, target_y_pred)
        print("\nMean Squared Error on Target Domain =", mse)
data = pd.read_csv('Periodic_Table.csv')
domain_adaptation = DomainAdaptation(data)
domain_adaptation.train_test_split()
domain_adaptation.preprocess_data()
domain_adaptation.fit_source_model()
domain_adaptation.evaluate_on_target(data)
