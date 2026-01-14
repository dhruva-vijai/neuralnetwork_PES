import os
os.environ["OMP_NUM_THREADS"] = "14"
os.environ["MKL_NUM_THREADS"] = "1" 
import random
import numpy as np
import math
import holder as hold
import tensorflow as tf

seed = 42

os.environ['PYTHONHASHSEED'] = str(seed)          # Sets Python hash seed
random.seed(seed)                                 # Python built-in RNG seed
np.random.seed(seed)                              # NumPy RNG seed
tf.random.set_seed(seed)                          # TensorFlow RNG seed

#generating data

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

#normalising data



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
    tf.keras.layers.Dense(100, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),    #0.08 rmse
    tf.keras.layers.Dense(1)
])
'''

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),  
    tf.keras.layers.Dense(1)
])

optimiser = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-5)
model.compile(optimizer=optimiser, loss='mse', metrics=['mse'])




model.fit(x_train_norm,y_train, epochs=400, batch_size=16)
'''

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(100, activation='relu'), - this is interesting 0.04 but weird
    tf.keras.layers.Dense(100, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),  
    tf.keras.layers.Dense(1)
])

'''
#testing
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


results = model.evaluate(x_test_norm, y_test, verbose=0)

loss = results[0]
mse = results[1]

# Print loss and mse
print(f"Loss: {loss:.4f}")
print(f"MSE: {mse:.4f}")

# Compute RMSE from MSE
rmse = mse**0.5
print(f"RMSE: {rmse:.4f}")

y_pred=model.predict(xpredict_norm)



    

    
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
