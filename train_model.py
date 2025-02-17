import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Dataset Paths
TRAINING_DIR = 'D:/desktop/rice_dataset3/train'
VALIDATION_DIR = 'D:/desktop/rice_dataset3/validation'

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=30, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, target_size=(150, 150), batch_size=32, class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=(150, 150), batch_size=32, class_mode='categorical')

# Load DenseNet121 Pretrained Model
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze base model layers

# Add Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_generator, validation_data=validation_generator, epochs=10, verbose=1)

# Save the Model
model_save_path = "rice_disease_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

