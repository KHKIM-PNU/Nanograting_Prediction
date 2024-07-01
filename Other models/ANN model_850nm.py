import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time

# Start the timer
start_time = time.time()

# 1. data
# load data
df = pd.read_csv("216_850nm.csv")
names = ('Pitch', 'Duty', 'Height')
x = df.loc[:, names]
y = df.loc[:, 'T']

print('train x:', x)
print('train y:', y)

df1 = pd.read_csv("test set_850nm.csv")
test_x = df1.loc[:, names]
test_y = df1.loc[:, 'T']

print('test x:', test_x)
print('test y:', test_y)

# split data(train:valid=8:2)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, random_state=1)

# 2. organize model
# Train & Validation MLP model
model = Sequential()
model.add(Dense(216, activation='swish', input_dim=3))
model.add(Dense(36, activation='swish'))
model.add(Dense(36, activation='swish'))
model.add(Dense(6, activation='swish'))
model.add(Dense(1, activation='linear'))

print(model.summary())

# 3. train
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(train_x, train_y, epochs=1000, batch_size=1, validation_data=(valid_x, valid_y))

train_predict_y = model.predict(x)
df2 = pd.DataFrame(model.predict(x))
print('predicted_T(train):', df2)
df2.to_csv('predict results(5).csv', index=False)

# 3(+) save model
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history(5).csv'
with open(hist_csv_file, mode='w', newline='') as f:
    hist_df.to_csv(f)

# 4. eval, pred
loss, mse = model.evaluate(test_x, test_y, batch_size=1)
print('loss(test):', mse)

predict_y = model.predict(test_x)
print('predicted_y(test):', predict_y)

# train MSE
mse_train_y = mean_squared_error(y, model.predict(x))
print('mse(train):', mse_train_y)

# test MSE
mse_predict_y = mean_squared_error(test_y, predict_y)
print('mse(test):', mse_predict_y)

# RMSE
def RMSE1(y, train_predict_y):
    return np.sqrt(mean_squared_error(y, train_predict_y))

print('RMSE(train):', RMSE1(y, train_predict_y))

def RMSE2(test_y, predict_y):
    return np.sqrt(mean_squared_error(test_y, predict_y))

print('RMSE(test):', RMSE2(test_y, predict_y))

# R2
r2_train_predict_y = r2_score(y, train_predict_y)
print('R2(train):', r2_train_predict_y)

r2_predict_y = r2_score(test_y, predict_y)
print('R2(test) : ', r2_predict_y)

# End the timer
end_time = time.time()

# Calculate the total training time
total_time = end_time - start_time

# Print the total training time
print("Total training time:", total_time, "seconds")
#5. Graph

# Train_graph
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist)
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss(mse)')
    # plt.plot(hist['epoch'], hist['loss'], label='Test')
    plt.plot(hist['epoch'], hist['loss'], label='Train')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val')
    # plt.xlim([-1,10])

    #plt.ylim([0, 200])
    plt.title('Loss of Train & Validation')
    # plt.title('Loss of Test')
    plt.legend()
    #plt.show()

plot_history(history)

def plot_history2(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist)
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss(mse)')
    # plt.plot(hist['epoch'], hist['loss'], label='Test')
    plt.plot(hist['epoch'], hist['loss'], label='Train')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val')
    plt.xlim([900,1000])
    plt.ylim([0, 2])
    plt.title('Loss of Train & Validation')
    # plt.title('Loss of Test')
    plt.legend()
    plt.show()

plot_history2(history)

