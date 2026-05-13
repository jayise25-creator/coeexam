# app.py

import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

st.set_page_config(page_title="Animal Detector", layout="centered")

st.title("🐾 Animal Detector using Teachable Machine")

st.write("Upload an animal image and detect the animal.")

# Load Teachable Machine model
model = load_model("keras_model.h5", compile=False)

# Load labels
class_names = open("labels.txt", "r").readlines()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Animal"):

        # Resize image
        image = image.resize((224, 224))

        # Convert to array
        image_array = np.asarray(image)

        # Normalize image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Prepare input
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)

        index = np.argmax(prediction)

        class_name = class_names[index]

        confidence_score = prediction[0][index]

        st.subheader("Prediction")

        st.write(f"Animal: {class_name[2:].strip()}")

        st.write(f"Confidence: {round(confidence_score * 100, 2)}%")
