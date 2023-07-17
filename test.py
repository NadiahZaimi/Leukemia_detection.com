import pandas as pd
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from random import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Set the path to the dataset directory
DATASET_DIR = "static/dataset"

# Preprocessing
def get_data(directory):
    X = []
    y = []
    classes = ['NORMAL', 'CANCER']

    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        class_label = classes.index(class_name)

        for file in tqdm(os.listdir(class_dir)):
            if not file.startswith('.'):
                img_path = os.path.join(class_dir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (150, 150))
                    X.append(img)
                    y.append(class_label)

    X = np.array(X)
    y = np.array(y)
    return X, y

X, y = get_data(os.path.join(DATASET_DIR, 'train'))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to categorical
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Define callbacks
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
# Train the model
batch_size = 32
epochs = 20

#classification
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, y_test), callbacks=[lr_reduce, checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot the accuracy and loss curves
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix
pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1)
y_true = np.argmax(y_test,axis = 1)

cnf = confusion_matrix(y_true, pred )
np.set_printoptions(precision=2)
cnf

from sklearn.metrics import accuracy_score
print ('Accuracy Score :',accuracy_score(y_true, pred))

from sklearn.metrics import classification_report
print(classification_report(y_true, pred))

from sklearn.metrics import precision_recall_curve

precision , recall , thresolds = precision_recall_curve(y_true, pred)

precision
recall
thresolds

# Generate predictions
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate and plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), show_absolute=True, show_normed=True)
plt.show()

from keras.models import load_model
from PIL import Image
import numpy as np
# Load the trained model
model = load_model('best_model.h5')
"""# Load and preprocess the test image
img_path = 'cancer/val/Cancer/_2_4392.jpeg'
img = Image.open(img_path)
img = img.resize((150, 150))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Make predictions
probs = model.predict(img)
predicted_class = np.argmax(probs)

# Map the predicted class index to the class label
class_labels = ['NORMAL', 'CANCER']
predicted_label = class_labels[predicted_class]

# Display the predicted label
print("Predicted Label:", predicted_label)"""

