import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Data Acquisition
df = pd.read_csv("216_850nm.csv")

# set color
colors = plt.cm.tab20(np.linspace(0, 1, 216)[0:len(df.label.unique())])
color_dic = {label: color for label, color in zip(df.label.unique(), colors)}
df['color'] = df.label.map(color_dic)

### Machine Learning Fit
# Make a list of variables for machine learning
names = ('Pitch', 'Duty', 'Height')
variables = df.loc[:, names]
T=df.loc[:,'T']

# Define a Pipeline that scales the data and applies the model
reg_T = Pipeline([('scl', StandardScaler()),
                    ('clf', svm.SVR(kernel='rbf', gamma=0.35, C=3, epsilon=0.1))]) #use best hyperparams

# Fit the variables to the T
reg_T.fit(variables, T)
# Get the predicted T from the model and save it to the DataFrame
df1=df['T_pred_svm'] = reg_T.predict(variables)

print(df['T_pred_svm'])
print(df['T'])

# #save T_pred
#df1.to_csv('T_pred_svm.csv', index=true, header=true)

## error function
#df.to_csv('850nm_prediction data.csv', columns=['Pitch', 'Duty', 'Height', 'T', 'T_pred_svm'], index='false')
print('MSE(train):', np.mean(np.square(T-reg_T.predict(variables))))
print('MAE:', np.mean(np.abs(T-reg_T.predict(variables))))
print("MAPE:", np.mean(np.abs((T-reg_T.predict(variables))/T))*100)
print('score:', reg_T.score(variables, T))

# test
df_test = pd.read_csv("Figure_2(h)_test set.csv")
variables2 = df_test.loc[:,names]
T2=df_test.loc[:,'T']
print(variables2)
Test_y=reg_T.predict(variables2)
print(Test_y)
print('MSE(test):', np.mean(np.square(T2-Test_y)))

# make a plot of the real values vs the predicted
fig, ax1 = plt.subplots(1, 1, clear=True, num="T_pred", figsize=(14,10))
for label, data in df.groupby('label'):
    plt.plot('T', 'T_pred_svm', 'o', color=data['color'].iloc[0], data=data, label=label)

plt.legend(loc=(1.05, 0.0), ncol=5)
plt.plot([90, 93], [90, 93], ls="--", c=".3")
ax1.set_ylabel('Predicted T(%)')
ax1.set_xlabel('Measured T(%)')
plt.tight_layout()

### Plot slices of the 3D fit with value maps / Contour map plot
var_n = 5
v_len = 6
vs = np.array([200, 240, 280, 320, 360, 400]) # Pitch
# vs = np.array([10, 25, 40, 55])

x_len, y_len = 100, 100
xs = np.linspace(0.32, 0.88, x_len) # Duty
ys = np.linspace(160, 440, y_len) # Height
vi, xi, yi = names

fig, axs = plt.subplots(nrows=1, ncols=v_len, sharex=True, sharey=True,
                        clear=True, num='rbf T plot', figsize=(18, 4))

# Slice through the "Pitch" direction
for ax, v in zip(axs, vs):

    xm, ym = np.meshgrid(xs, ys)
    vm = v * np.ones_like(xm)
    r = np.c_[vm.flatten(), xm.flatten(), ym.flatten()]

    # Compute the values from the fit
    c = reg_T.predict(r).reshape(x_len, y_len)
    # print(c)

    # Make a contour map
    cmap = ax.contour(xs, ys, c, vmin=91.5, vmax=93, cmap='gray_r')

    plt.clabel(cmap, inline=1, fontsize=12)

    # Make a value map
    pmap = ax.pcolormesh(xs, ys, c, shading='gouraud',
                         vmin=90, vmax=93, cmap='viridis')

    # Plot the experimental points
    for label, data in df.query('Pitch==@v').groupby('label'):
        ax.plot('Duty', 'Height', 'o', color=data['color'].iloc[0],
                data=data.iloc[0], mec='k', mew=0.5, label=label)

    ax.set_ylabel(f'{yi}')
    ax.set_xlabel(f'{xi}\n{vi}={v:.2f}')

plt.tight_layout()
plt.colorbar(pmap, ax=axs, fraction=0.04)
plt.show()
