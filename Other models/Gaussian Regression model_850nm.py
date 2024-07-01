import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# load CSV
data = pd.read_csv("216_850nm.csv")

# set x, y
X = data[['Pitch', 'Duty', 'Height']].values
y = data['T'].values

# set grid
param_grid = {
    'kernel__length_scale': [0.1, 1.0, 5.0, 10.0],
    'alpha': [0.001, 0.01, 0.1, 1.0]
}

# Gaussian Process Regressor
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 100000.0))
model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)

# grid search
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X, y)

# get best hyperparams & model
best_hyperparams = grid_search.best_params_
best_model = grid_search.best_estimator_

# train
best_model.fit(X, y)

# predict
data['y_pred'] = best_model.predict(X)
print(best_hyperparams)
print(data['y_pred'])

# save prediction
#data['y_pred'].to_csv('GR_850nm_pred(alpha 0.001, kenel length 0.1).csv', index=False)

# set color
colors = plt.cm.viridis(np.linspace(0, 1, len(data['label'].unique())))
color_dict = {label: color for label, color in zip(data['label'].unique(), colors)}
data['color'] = data['label'].map(color_dict)

# graph
fig, ax1 = plt.subplots(1, 1, clear=True, num="y_pred", figsize=(10, 6))
for label, group in data.groupby('label'):
    ax1.plot(group['T'], group['y_pred'], 'or', color=group['color'].iloc[0], label=label)

plt.legend(loc=(1.05, 0.0), ncol=6)
plt.plot([min(data['T']), max(data['T'])], [min(data['T']), max(data['T'])], ls="--", c=".3")
ax1.set_ylabel('Predicted T')
ax1.set_xlabel('Simulated T')
plt.tight_layout()
plt.show()

# MSE
mse = mean_squared_error(data['T'], data['y_pred'])
print(f"MSE: {mse:.4f}")

# R-squared
r2 = r2_score(data['T'], data['y_pred'])
print(f"R-squared: {r2:.4f}")
