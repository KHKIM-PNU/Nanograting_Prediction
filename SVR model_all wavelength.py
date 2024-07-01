import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

### load data
dflist = [pd.read_csv("216_400nm.csv"), pd.read_csv("216_450nm.csv"), pd.read_csv("216_500nm.csv"), pd.read_csv("216_550nm.csv"), pd.read_csv("216_600nm.csv"),
          pd.read_csv("216_650nm.csv"), pd.read_csv("216_700nm.csv"), pd.read_csv("216_750nm.csv"), pd.read_csv("216_800nm.csv"), pd.read_csv("216_850nm.csv"),
          pd.read_csv("216_900nm.csv"), pd.read_csv("216_950nm.csv"), pd.read_csv("216_1000nm.csv")]
reg_Tlist = []

for i in range(len(dflist)):
    print("number : "+ str(i))
    # set color
    colors = plt.cm.tab20(np.linspace(0, 1, 216)[0:len(dflist[i].label.unique())])
    color_dic = {label: color for label, color in zip(dflist[i].label.unique(), colors)}
    dflist[i]['color'] = dflist[i].label.map(color_dic)

    ### train
    names = ('Pitch', 'Duty', 'Height')
    variables = dflist[i].loc[:, names]
    T = dflist[i].loc[:, 'T']

    reg_Tlist.append(Pipeline([('scl', StandardScaler()),
                               ('clf', svm.SVR(kernel='rbf', gamma=0.15, C=20, epsilon=0.1))]))

    reg_Tlist[i].fit(variables.values, T)
    df1 = dflist[i]['T_pred_svm'] = reg_Tlist[i].predict(variables)

    print(dflist[i]['T_pred_svm'])
    print(dflist[i]['T'])

    print('MSE(train):', np.mean(np.square(T - reg_Tlist[i].predict(variables))))
    print('score:', reg_Tlist[i].score(variables, T))

### Contour map plot
var_n = 5
v_len = 6
vs = np.array([200, 240, 280, 320, 360, 400])

x_len, y_len = 100, 100
xs = np.linspace(0.32, 0.88, x_len)
ys = np.linspace(160, 440, y_len)

fig, axs = plt.subplots(nrows=1, ncols=v_len, sharex=True, sharey=True,
                        clear=True, num='rbf T plot', figsize=(14, 3))

clist = []
for ax, v in zip(axs, vs):
    xm, ym = np.meshgrid(xs, ys)
    vm = v * np.ones_like(xm)
    r = np.c_[vm.flatten(), xm.flatten(), ym.flatten()]

    c1 = (reg_Tlist[0].predict(r).reshape(x_len, y_len))
    c2 = (reg_Tlist[1].predict(r).reshape(x_len, y_len))
    c3 = (reg_Tlist[2].predict(r).reshape(x_len, y_len))
    c4 = (reg_Tlist[3].predict(r).reshape(x_len, y_len))
    c5 = (reg_Tlist[4].predict(r).reshape(x_len, y_len))
    c6 = (reg_Tlist[5].predict(r).reshape(x_len, y_len))

    c = [[(c11 + c22 + c33 + c44 + c55 + c66) / 6 for c11, c22, c33, c44, c55, c66 in zip(c1[i], c2[i], c3[i], c4[i], c5[i], c6[i])] for i in range(len(c1))]

    cmap = ax.contour(xs, ys, c, vmin=90, vmax=91, cmap='gray_r')
    plt.clabel(cmap, inline=1, fontsize=10)

    pmap = ax.pcolormesh(xs, ys, c, shading='gouraud', vmin=90, vmax=91.6, cmap='viridis')



ax.set_xlabel('')
ax.set_ylabel('')
ax.axis('off')

plt.tight_layout()
plt.show()
