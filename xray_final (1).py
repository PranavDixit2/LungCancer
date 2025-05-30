# -*- coding: utf-8 -*-
"""XRay_Final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WPakXPtF90F7phglLMtZDY1xRnP6cryc
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained model
model_path = "/content/drive/MyDrive/lung_cancer_model.keras"
model = tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to match model input
    image_array = img_to_array(image)  # Convert to array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)  # Preprocess for ResNet50
    return image_array

# Streamlit app layout
st.title("Lung Cancer Detection")
st.write("Upload a chest X-ray image to detect lung cancer.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    prediction_label = "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer Detected"

    # Display prediction
    st.write(f"Prediction: {prediction_label}")
    st.write(f"Prediction Probability: {prediction[0][0]:.2f}")

# Run the app
if __name__ == "__main__":
    st.run()