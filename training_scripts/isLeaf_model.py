import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

# Dataset Paths
train_dir = "/home/raj/Desktop/AI/AgriScanAI/Dataset/LNL_Dataset/train"
test_dir = "/home/raj/Desktop/AI/AgriScanAI/Dataset/LNL_Dataset/test"

# Parameters
img_size = (256,256)
batch_size = 16
num_classes = 2
epochs = 12

# Data Loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels = "inferred",
    label_mode = "binary",
    image_size = img_size,
    batch_size = batch_size,
    shuffle = True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels = "inferred",
    label_mode = "binary",
    image_size = img_size,
    batch_size = batch_size,
    shuffle = False
)

#Data Augmentation

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Normalization
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Auto-tunning 

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(300).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Model

model = keras.Sequential([
    layers.Input(shape=(256,256,3)),

    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')


])

model.compile(
    optimizer= keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()


history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
)


# Save Model
model.save("lnl_model.h5")
print("✅ Model saved successfully!")



#plotting training and validation accuracy and loss

def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14,5))

    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.scatter(np.argmax(val_acc)+1, max(val_acc), color='green', 
                label=f'Best Val Acc: {max(val_acc):.2f}', zorder=5)
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()


    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.scatter(np.argmin(val_loss)+1, min(val_loss), color='green',
                label=f'Best Val Loss: {min(val_loss):.4f}', zorder=5)
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()

# Call it after training
plot_training(history)


