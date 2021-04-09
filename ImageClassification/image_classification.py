# Image classification from scratch
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
 
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
#from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf 
import sys
import PIL
import pprint

#from google.colab import drive
#drive.mount('/content/drive')

print("\n\n\n\n\n");
#print("------- THREADS -------\n")
#tf.config.threading.set_inter_op_parallelism_threads(1)
print("------- STRATEGY -------\n")
mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/device:XLA_CPU:0'])
#tf.device("/device:XLA_CPU:0")
print("-----------------------\n")
print("\n\n\n\n\n");

AUTOTUNE = tf.data.experimental.AUTOTUNE
#BATCH_SIZE = 16 * strategy.num_replicas_in_sync
BATCH_SIZE = 16
IMAGE_SIZE = [200, 200]
EPOCHS = 50
#DATA_PATH = "/content/drive/My Drive/BIA/chest_xray/chest_xray"
DATA_PATH = './chest_xray';
os.listdir(DATA_PATH)
 
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2,
)
 
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255,
)
 
train_gen = train_datagen.flow_from_directory(
    '{}/train'.format(DATA_PATH),
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)
validation_gen = train_datagen.flow_from_directory(
    '{}/train'.format(DATA_PATH),
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)
test_gen = test_datagen.flow_from_directory(
    '{}/test'.format(DATA_PATH),
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    batch_size=BATCH_SIZE,
    class_mode='binary',
)
 
num_classes = 2
 
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(rate = 0.5))
model.add(Dense(512, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
 
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)
 
model.summary()


def plotresults(data, epochs, filename):
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, data['accuracy'],     label='Training Accuracy')
    plt.plot(epochs_range, data['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
     
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, data['loss'],     label='Training Loss')
    plt.plot(epochs_range, data['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(filename)

iterations = 10
t_ls = np.repeat(np.nan, iterations)
df = pd.DataFrame()

data = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

for iter in range(1, (iterations+1)): 
    start = time.time()
    train_val = model.fit(train_gen, epochs=5, validation_data=validation_gen, use_multiprocessing=True)
    end = time.time()
    t_ls[iter-1] = end - start
    df[iter-1] = model.evaluate(test_gen)
    for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
        data[key] += train_val.history[key]
    plotresults(data, iter*5, 'results{}.png'.format(iter*5));


# Save files
df_output = df.T
df_output = df_output.rename(columns = {0: 'loss', 1: 'accuracy', 2: 'precision', 3: 'recall', 4: 'AUC'})
df_output['epochs'] = range(5, 55, 5)
df_output['train_time(s)'] = t_ls
df_output.to_csv("./table1.csv")

