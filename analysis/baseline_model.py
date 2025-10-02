from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np



data = pd.read_csv('CCES2012_CSVFormat_NEW.csv')

X = data.drop(['CC12'], axis=1)
y = data['CC12']

# Linear model with county features 
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_model.score(X, y)
"""
R2 = ~0.45
"""

## Baseline with non-standardized features 

ridge = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100, 200, 300], 'fit_intercept': [True, False]}
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5)
grid_search.fit(X, y)
print("Best parameters for baseline Ridge Regression: ", grid_search.best_params_) 
print("R2 score: ", grid_search.best_score_)

"""
Best parameters for baseline Ridge Regression:  {'alpha': 100}
R2 score:  0.4185151366506159
"""


## Baseline with standardized features 

sc = StandardScaler()
X_c = sc.fit_transform(X)

ridge_2 = Ridge()

grid_search_sc = GridSearchCV(ridge_2, param_grid, cv=5)
grid_search_sc.fit(X_c, y)
print("Best parameters for baseline Ridge Regression with standardized features: ", grid_search_sc.best_params_) 
print("R2 score with standardized features: ", grid_search_sc.best_score_)


"""
Best parameters for baseline Ridge Regression with standardized features:  {'alpha': 200, 'fit_intercept': True}
R2 score with standardized features:  0.4184565576233914
"""


rfr = RandomForestRegressor(max_features="sqrt", n_jobs=-1, verbose=1)

rfr_param_grid = {
    "n_estimators": np.linspace(50, 200, 5)
}


rfr_grid_search = GridSearchCV(rfr, param_grid=rfr_param_grid, n_jobs=-1, verbose=1, cv=5)
rfr_grid_search.fit(X, y)

rfr_grid_search.best_params_
rfr_grid_search.best_score_

"""
{'n_estimators': 150}
R2 score: 0.41939151976690575
"""



xgbr = xgb.XGBRegressor()

xgbr_param_grid={
    "n_estimators":[500],
    "learning_rate": [0.025],
    "max_depth": [15]
}

grid_search_xgbr = GridSearchCV(xgbr, xgbr_param_grid, n_jobs = 7, verbose=1, cv=5)
grid_search_xgbr.fit(X, y)
grid_search_xgbr.best_params_
grid_search_xgbr.best_score_

"""
{‘n_estimators’: 200, ‘learning_rate’: 0.025’, ‘max_depth’: 15} 
R2 score: 0.4210396409034729
"""
