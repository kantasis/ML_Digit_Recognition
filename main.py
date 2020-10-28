import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Conv2D, Input, Dense, Activation, Flatten

# Prepare
np.random.seed(0)

# Define the variables
filename = "input/train.csv"
trainingFraction_float = 0.9
pixelRows = 28

# Import the data
dataset_df = pd.read_csv(filename)
N, M = dataset_df.shape
dataset_df = dataset_df.iloc[np.random.permutation(N)]

# Preprocess the data

trainingN_cnt = int( N * trainingFraction_float)
validationN_cnt = N - trainingN_cnt

trainX_arr = np.asarray(dataset_df.iloc[:trainingN_cnt,1:]).reshape([trainingN_cnt,28,28,1]) # taking all columns expect column 0
trainY_arr = np.asarray(dataset_df.iloc[:trainingN_cnt,0]).reshape([trainingN_cnt,1]) # taking column 0

valX_arr = np.asarray(dataset_df.iloc[trainingN_cnt:,1:]).reshape([validationN_cnt,pixelRows,pixelRows,1])
valY_arr = np.asarray(dataset_df.iloc[trainingN_cnt:,0]).reshape([validationN_cnt,1])

maxValue_float = trainX_arr.max().max()
minValue_float = trainX_arr.min().min()

trainX_arr = (trainX_arr - minValue_float)/(maxValue_float - minValue_float)
valX_arr = (valX_arr - minValue_float)/(maxValue_float - minValue_float)


# Visualize the data

figureRows_cnt = 4
figureCols_cnt = 4

figure = plt.figure(figsize=(2*figureRows_cnt,2*figureCols_cnt)) 

for i in range(figureRows_cnt*figureCols_cnt): 
    figure.add_subplot(figureRows_cnt,figureCols_cnt,i+1)
    plt.imshow(trainX_arr[i].reshape([pixelRows,pixelRows]),cmap="Blues") 
    plt.axis("off")
    plt.title(str(trainY_arr[i]), y=-0.15,color="green")


# Build the model

model = models.Sequential()
model.add(Conv2D(
    20,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(pixelRows, pixelRows, 1)
))
model.add(Conv2D(
   20, 
   kernel_size=(3, 3), 
   activation='relu')
)

model.add(Flatten())

model.add(Dense(10,activation="sigmoid"))

lRate_float = 0.001
lossFunction_str = "sparse_categorical_crossentropy"
metric_strList=['accuracy']
optimizer = tf.keras.optimizers.Adam(lr=lRate_float)

model.compile(optimizer, loss=lossFunction_str ,metrics=metric_strList)
model.summary()

# Fit the model

epoch_cnt = 20
batchSize_cnt = 256
history_1 = model.fit(
    trainX_arr,
    trainY_arr,
    batch_size=batchSize_cnt,
    epochs=epoch_cnt,
    validation_split = 0.2
)



figure = plt.figure(figsize=(20,7))
figure.add_subplot(121)

plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set
plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

plt.title("Accuracy Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

#Adding Subplot 1 (For Loss)
figure.add_subplot(122)

plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set

plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()















































