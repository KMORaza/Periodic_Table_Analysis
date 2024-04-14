import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names]
class StackingModel:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
    def fit_base_models(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
    def predict_base_models(self, X):
        predictions = pd.DataFrame()
        for i, model in enumerate(self.base_models):
            predictions[f'model_{i+1}'] = model.predict(X)
        return predictions
    def fit(self, X, y):
        self.fit_base_models(X, y)
        meta_features = self.predict_base_models(X)
        self.meta_model.fit(meta_features, y)
    def predict(self, X):
        meta_features = self.predict_base_models(X)
        return self.meta_model.predict(meta_features)
periodic_table = pd.read_csv("Periodic_Table.csv")
periodic_table_cleaned = periodic_table.dropna()
X = periodic_table_cleaned.drop(columns=["Boiling Point"])  
y = periodic_table_cleaned["Boiling Point"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
base_models = [
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42)
]
meta_model = LinearRegression()
stacking_model = StackingModel(base_models, meta_model)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error = {mse}")
