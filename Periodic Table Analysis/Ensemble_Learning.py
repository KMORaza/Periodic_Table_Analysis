import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
class PeriodicTableEnsemble:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.data.dropna(inplace=True)  
        self.X = self.data.drop(columns=['Boiling Point'])
        self.y = self.data['Boiling Point']  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.models = []
    def train_model(self, n_estimators=100, max_depth=None):
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models.append(model)
        return model
    def predict(self, X):
        predictions = []
        for model in self.models:
            prediction = model.predict(X)
            predictions.append(prediction)
        return predictions
    def evaluate(self):
        ensemble_predictions = self.predict(self.X_test)
        average_prediction = sum(ensemble_predictions) / len(ensemble_predictions)
        mse = mean_squared_error(self.y_test, average_prediction)
        return mse
ensemble = PeriodicTableEnsemble("Periodic_Table.csv")
ensemble.train_model()
mse = ensemble.evaluate()
print("Mean Squared Error =", mse)
