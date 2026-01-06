import os
os.environ["OMP_NUM_THREADS"] = "14"
os.environ["MKL_NUM_THREADS"] = "1" 

import numpy as np
import math
import holder as hold
import scipy
import sklearn
import os
import random
import numpy as np
import tensorflow as tf


print("generating data")

d=hold.give()
x_train=[]
y_train=[]

for i in d:
    x_train.append(i)
    y_train.append(d[i])


x_train=np.array(x_train) #(196,2)
y_train=np.array(y_train).reshape(-1,1)


xpredict=[]
firstin=x_train[0][0]
firstfin=x_train[-1][0]
secin=x_train[0][1]
secfin=x_train[-1][1]

xpredict=[]#ML-PES - replace with higher resolution grid(in this case its testing data)
for i in np.linspace(firstin,firstfin,100):
    for j in np.linspace(secin,secfin,100):
        xpredict.append([i,j])

xpredict=np.array(xpredict) #(196,2)

print("normalising data")



X_train=x_train.astype(np.float64)
Xpredict=xpredict.astype(np.float64)


mean=X_train.mean(axis=0)

std=X_train.std(axis=0,ddof=0)

x_train_norm=(X_train-mean)/(std)
xpredict_norm=(Xpredict-mean)/(std)



#No Validation Neural Network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, Input

'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),    #last 64 gives0.13
    tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])


model.fit(x_train_norm,y_train, epochs=400, batch_size=32)#initial ep 100(0.58),300(0.26),400(0.18)
y_pred=model.predict(xpredict_norm)

'''

print("building net")

import tensorflow as tf
from tensorflow.keras import layers, regularizers
import keras_tuner
import numpy as np

def build_model(hp): #hyperparams - /layers,/units,/activation,/dropout,/batch size,/epoch,/learning rate
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hp.Choice("units1", [32,64,96,128,224,256]),activation=hp.Choice("activation1",['relu','elu','sigmoid','tanh'])))
    model.add(tf.keras.layers.Dropout(hp.Choice("dropout1",[0.2,0.4,0.6])))
    model.add(tf.keras.layers.Dense(hp.Choice("units2", [32,64,96,128,224,256]),activation=hp.Choice("activation2",['relu','elu','sigmoid','tanh'])))
    model.add(tf.keras.layers.Dropout(hp.Choice("dropout2",[0.2,0.4,0.6])))
    model.add(tf.keras.layers.Dense(hp.Choice("units1", [32,64,96,128,224,256]),activation=hp.Choice("activation1",['relu','elu','sigmoid','tanh'])))
    model.add(tf.keras.layers.Dropout(hp.Choice("dropout1",[0.2,0.4,0.6])))
    model.add(tf.keras.layers.Dense(hp.Choice("units2", [32,64,96,128,224,256]),activation=hp.Choice("activation2",['relu','elu','sigmoid','tanh'])))
    model.add(tf.keras.layers.Dropout(hp.Choice("dropout2",[0.2,0.4,0.6])))

    model.add(tf.keras.layers.Dense(1))

    l_rate=hp.Choice("learning_rate",[1e-2,1e-3,1e-4])
    ochoice=hp.Choice("Optimiser",['adam'])
    if ochoice=='adam':
        optimiser=tf.keras.optimizers.Adam(learning_rate=l_rate)
    model.compile(optimizer=optimiser, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model


tuner = keras_tuner.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=300,
    num_initial_points=40,
    overwrite=True)

stop_early=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=20,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
    )
    
tuner.search(x_train_norm,y_train,batch_size=64,epochs=300, validation_split=0.2, callbacks=[stop_early])


ideal = tuner.get_best_models()[0]




print("test")

d=hold.givetest()
x_test=[]
y_test=[]

for i in d:
    x_test.append(i)
    y_test.append(d[i])


x_test=np.array(x_test)
y_test=np.array(y_test).reshape(-1,1)
X_test=x_test.astype(np.float64)
x_test_norm=(X_test-mean)/(std)


results = ideal.evaluate(x_test_norm, y_test, verbose=0)

loss = results[0]
mse = results[1]

# Print loss and mse
print(f"Loss: {loss:.4f}")
print(f"MSE: {mse:.4f}")

# Compute RMSE from MSE
rmse = mse**0.5
print(f"RMSE: {rmse:.4f}")

y_pred=ideal.predict(xpredict_norm)                  

y_pred=ideal.predict(xpredict_norm)
    

    
#Their RMSE = 3.013e-03 --> num. to beat ; MSE = 9e-06

#plotting final predictions

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


xnp = np.array(xpredict)
z = y_pred.flatten()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(xnp[:, 0], xnp[:, 1], z, c=z, cmap='viridis')


xi = np.linspace(xnp[:, 0].min(), xnp[:, 0].max(), 50)
yi = np.linspace(xnp[:, 1].min(), xnp[:, 1].max(), 50)
xi, yi = np.meshgrid(xi, yi)

from scipy.interpolate import griddata
zi = griddata((xnp[:, 0], xnp[:, 1]), z, (xi, yi), method='cubic')

ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.6)

ax.set_xlabel('CH Bond Length')
ax.set_ylabel('HH Bond Distance')
ax.set_zlabel('Prediction')
ax.set_title('3D Surface Plot of Predictions')

plt.show()
