# app.py

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Animal Detector", layout="centered")

st.title("🐶 Animal Detector App")

st.write("Upload an image and the AI will try to detect the animal.")

# Load MobileNet model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an Animal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Show image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Detect Animal"):

        # Resize image
        img = image.resize((224, 224))

        # Convert image to array
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess image
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Prediction
        predictions = model.predict(img_array)

        # Decode prediction
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
            predictions,
            top=3
        )[0]

        st.subheader("Prediction Results")

        for i, pred in enumerate(decoded):
            animal_name = pred[1].replace("_", " ").title()
            confidence = round(pred[2] * 100, 2)

            st.write(f"{i+1}. {animal_name} — {confidence}% confidence")
