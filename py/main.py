import tensorflow as tf
from keras import datasets, layers, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
res = 'C:/Users/dines/OneDrive/Desktop/Github/Cats and dogs classifier/train/'
img_extension = ['png', 'jpg']
files = []
[files.extend(glob.glob(res + '*.' + e)) for e in img_extension]
pet_img = np.asarray([cv2.imread(file) for file in files])

# Create labels
labels = []
for img in os.listdir(res):
    if img.startswith('cat'):
        labels.append(0)
    else:
        labels.append(1)

# Convert to numpy arrays
x = pet_img
y = np.asarray(labels)
x_resized = np.array([cv2.resize(img, (32, 32)) for img in x])
# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_resized, y, test_size=0.2, random_state=2)
x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 255.0
x_scaled = x_resized / 255.0


# Model definition
cnn = models.Sequential([
    layers.Conv2D(filters=32, activation='relu',kernel_size=(3,3), input_shape = (32,32,3) ),# 
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64, activation='relu',kernel_size=(3,3) ),
    layers.MaxPooling2D((2,2)),
    #dense
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax') # Changed to softmax for classification
])

print(cnn.summary())

# Compile the model
cnn.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

# Train the model
cnn.fit(x_train_scaled, y_train, epochs=50, validation_split=0.2)  # Include validation split

# Evaluate the model
res = cnn.predict(x_test_scaled)
res_classes = np.argmax(res, axis=1)
print('Test Accuracy:', accuracy_score(y_test, res_classes))

# Save the model
with open('Model.sav', 'wb') as pick_in:
    pickle.dump(cnn, pick_in)
