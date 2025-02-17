import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Load the trained model
MODEL_PATH = "rice_disease_model.h5"
model = load_model(MODEL_PATH)

# Class Labels
class_labels = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']

# Severity Levels & Prescriptions
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

# Function to Predict Disease
def predict_disease(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    disease = class_labels[class_idx]
    severity = severity_levels[disease]
    prescription = prescriptions[disease]
    
    return disease, severity, prescription

# Streamlit UI
st.title('🌾 Rice Plant Disease Prediction')
st.write('Upload an image of a rice leaf to predict its disease and receive treatment advice.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image_path = f"uploaded_images/{uploaded_file.name}"
    
    # Ensure directory exists
    os.makedirs("uploaded_images", exist_ok=True)
    
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(image_path, caption='Uploaded Image.', use_column_width=True)
    
    disease, severity, prescription = predict_disease(image_path)
    
    st.subheader(f"Prediction: {disease}")
    st.write(f"**Severity Level:** {severity}")
    st.write(f"**Prescription:** {prescription}")

