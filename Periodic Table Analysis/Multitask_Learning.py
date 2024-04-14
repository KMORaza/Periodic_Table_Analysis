import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
data = pd.read_csv("Periodic_Table.csv")
X = data.drop(columns=['Boiling Point', 'Density', 'Melting Point'])
y = data[['Boiling Point', 'Density', 'Melting Point']]
missing_rows = X.index.difference(y.index)
X.drop(index=missing_rows, inplace=True)
y.drop(index=missing_rows, inplace=True)
y.dropna(inplace=True)
X.dropna(inplace=True)
common_indices = X.index.intersection(y.index)
X = X.loc[common_indices]
y = y.loc[common_indices]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))  
])
param_grid = {
    'regressor__estimator__n_estimators': [50, 100, 150],
    'regressor__estimator__max_depth': [None, 10, 20],
}
scorer = make_scorer(rmse, greater_is_better=False)
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorer, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = -grid_search.best_score_
print("\nBest Parameters:", best_params)
print("Best RMSE =", best_score)
pipeline.set_params(**best_params)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
test_rmse = rmse(y_test, y_pred)
print("\nTest RMSE =", test_rmse)
