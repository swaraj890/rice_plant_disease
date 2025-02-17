import streamlit as st
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
