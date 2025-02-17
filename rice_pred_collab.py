import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from tensorflow.keras.utils import img_to_array, load_img

TRAINING_DIR = 'D:/desktop/rice_dataset3/train'  # Using forward slashes
VALIDATION_DIR = 'D:/desktop/rice_dataset3/validation'  # Using forward slashes


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

# Freeze the base model layers
base_model.trainable = False

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

# Evaluate the Model
eval_loss, eval_accuracy = model.evaluate(validation_generator, verbose=1)
print(f"Validation Accuracy: {eval_accuracy}")

# Calculate F1 Score and Confusion Matrix
y_true = validation_generator.classes
y_pred = np.argmax(model.predict(validation_generator), axis=-1)

f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1}")

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Training and Testing Loss Graphs
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Severity Levels and Prescriptions for each disease
severity_levels = {
    'Bacterial Leaf Blight': 'Extreme',
    'Brown Spot': 'Moderate',
    'Healthy': 'No Disease',
    'Leaf Blast': 'Extreme',
    'Leaf Scald': 'Moderate',
    'Narrow Brown Spot': 'Moderate'
}

prescriptions = {
    'Bacterial Leaf Blight': 'Use bactericides like Streptomycin and copper-based fungicides.',
    'Brown Spot': 'Apply nitrogen fertilizers and improve drainage in the field.',
    'Healthy': 'No action required.',
    'Leaf Blast': 'Apply fungicides such as Propiconazole or Mancozeb.',
    'Leaf Scald': 'Use resistant varieties and apply fungicides like Copper-based solutions.',
    'Narrow Brown Spot': 'Increase nitrogen fertilizer use and improve drainage.'
}

# Streamlit App for Image Upload and Prediction
def predict_disease(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    class_labels = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']
    disease = class_labels[class_idx]
    severity = severity_levels[disease]
    prescription = prescriptions[disease]
    
    return disease, severity, prescription

# Streamlit UI for Image Upload
st.title('Rice Plant Disease Prediction')
st.write('Upload an image of a rice leaf to predict its disease and receive severity level and treatment advice.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save the uploaded image and predict disease
    image_path = f"uploaded_images/{uploaded_file.name}"
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(image_path, caption='Uploaded Image.', use_column_width=True)
    
    disease, severity, prescription = predict_disease(image_path)
    st.write(f"Prediction: {disease}")
    st.write(f"Severity Level: {severity}")
    st.write(f"Prescription: {prescription}")
