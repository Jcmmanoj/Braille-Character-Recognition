"""
train_model_clean.py

Trains a CNN for Braille script recognition and saves the model for GUI use.

Requirements:
- Folder structure: ./dataa/<class_name>/*.jpg
- TensorFlow / Keras installed
"""

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========================
# Paths and Hyperparameters
# ========================
DATA_DIR = './dataa'  # folder with subfolders for each class
MODEL_SAVE_PATH = r"C:\Users\jcmma\OneDrive\Desktop\Code 1 project\trained_model_CNN.h5"

IMG_WIDTH, IMG_HEIGHT = 224, 224
EPOCHS = 30
BATCH_SIZE = 16

# ========================
# Load Dataset
# ========================
def load_dataset(path):
    files = []
    labels = []
    class_folders = sorted(glob(os.path.join(path, '*')))
    class_names = [os.path.basename(os.path.normpath(c)) for c in class_folders]
    num_classes = len(class_names)

    for idx, class_folder in enumerate(class_folders):
        imgs = glob(os.path.join(class_folder, '*'))
        files.extend(imgs)
        labels.extend([idx] * len(imgs))

    files = np.array(files)
    labels = to_categorical(np.array(labels), num_classes)
    return files, labels, class_names

train_files, train_targets, class_names = load_dataset(DATA_DIR)
num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")
print(f"Total images: {len(train_files)}")

# ========================
# Convert image paths to tensors
# ========================
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    tensors = [path_to_tensor(p) for p in img_paths]
    return np.vstack(tensors)

print("Processing images into tensors...")
train_tensors = paths_to_tensor(train_files).astype('float32') / 255.0

# ========================
# Define CNN model
# ========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========================
# Callbacks
# ========================
class EpochTimer(Callback):
    import timeit
    def on_train_begin(self, logs=None):
        self.start = self.timeit.default_timer()
    def on_train_end(self, logs=None):
        print(f"Training took {self.timeit.default_timer() - self.start:.2f} seconds")

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# ========================
# Train the model
# ========================
history = model.fit(
    train_tensors,
    train_targets,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, EpochTimer()],
    validation_split=0.1
)

# ========================
# Save model
# ========================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# ========================
# Plot training history
# ========================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
