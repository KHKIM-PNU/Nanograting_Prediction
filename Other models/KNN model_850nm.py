import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

# load data
df = pd.read_csv("216_850nm.csv")

# set color
colors = plt.cm.tab20(np.linspace(0, 1, 216)[0:len(df.label.unique())])
color_dic = {label: color for label, color in zip(df.label.unique(), colors)}
df['color'] = df.label.map(color_dic)

# Machine Learning Fit
# Make a list of variables for machine learning
names = ('Pitch', 'Duty', 'Height')
variables = df.loc[:, names]
T=df.loc[:,'T']
reg_T = neighbors.KNeighborsRegressor(n_neighbors=3)
reg_T.fit(variables, T)


# Get the predicted T from the model and save it to the DataFrame
df1=df['T_pred_KNN'] = reg_T.predict(variables)

np.savetxt('T_pred_KNN.csv',df1,delimiter=" ")

# print(df['T_pred_svm'])
# print(df['T'])
print(df['T_pred_KNN'])
print(df['T'])

print('MSE:', np.mean(np.square(T-reg_T.predict(variables))))
print('score:', reg_T.score(variables, T))

df_test = pd.read_csv("test set_850nm.csv")
variables2 = df_test.loc[:,names]
print(variables2)
Test_y=reg_T.predict(variables2)
print(Test_y)
# make a plot of the real values vs the predicted
# Increase gamma in the pipeline until the data just starts on tho lay
# on the line. If gamma is too high the data can be over fit
fig, ax1 = plt.subplots(1, 1, clear=True, num="T_pred", figsize=(14,7))
for label, data in df.groupby('label'):
    plt.plot('T', 'T_pred_KNN', 'o', color=data['color'].iloc[0], data=data, label=label)

plt.legend(loc=(1.05, 0.0), ncol=6)
# plt.autoscale(enable=False)
plt.plot([90, 93], [90, 93], ls="--", c=".3")
ax1.set_ylabel('Predicted T(%)')
ax1.set_xlabel('Measured T(%)')
plt.tight_layout()
plt.savefig('performance_KNN.png', facecolor='#eeeeee', edgecolor='black', format='png', dpi=300)
# plt.axis([0, 7, 0, 7])
# plt.show()

### Plot slices of the 3D fit with value maps
var_n = 5
v_len = 6
vs = np.array([200, 240, 280, 320, 360, 400])  # v is p(period)
x_len, y_len = 100, 100
xs = np.linspace(0.32, 0.88, x_len)
ys = np.linspace(160, 440, y_len)
vi, xi, yi = names

fig, axs = plt.subplots(nrows=1, ncols=v_len, sharex=True, sharey=True,
                        clear=True, num='rbf T plot', figsize=(13, 4))

# Slice through the "don_con" direction
for ax, v in zip(axs, vs):

    xm, ym = np.meshgrid(xs, ys)
    vm = v * np.ones_like(xm)
    r = np.c_[vm.flatten(), xm.flatten(), ym.flatten()]

    # Compute the values from the fit
    c = reg_T.predict(r).reshape(x_len, y_len)
    # print(c)

    # Make a contour map
    cmap = ax.contour(xs, ys, c, vmin=0, vmax=94, cmap='gray_r')

    plt.clabel(cmap, inline=1, fontsize=10)

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
plt.savefig('contourmap_KNN.png', facecolor='#eeeeee', edgecolor='black', format='png', dpi=300)
plt.show()
