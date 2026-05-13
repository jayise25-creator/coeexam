# app.py

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Page settings
st.set_page_config(page_title="Animal Detector", layout="centered")

st.title("🐾 Animal Detector")
st.write("Upload an animal image and detect the animal.")

# Load model
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

    # Show image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Detect button
    if st.button("Detect Animal"):

        # Resize image
        image = image.resize((224, 224))

        # Convert image to array
        image_array = np.asarray(image)

        # Normalize image
        normalized_image_array = (
            image_array.astype(np.float32) / 127.5
        ) - 1

        # Prepare data
        data = np.ndarray(
            shape=(1, 224, 224, 3),
            dtype=np.float32
        )

        data[0] = normalized_image_array

        # Prediction
        prediction = model.predict(data)

        # Get index
        index = np.argmax(prediction)

        # Get class name
        class_name = class_names[index]

        # Confidence score
        confidence_score = prediction[0][index]

        # Show result
        st.subheader("Prediction Result")

        st.success(
            f"Animal: {class_name[2:].strip()}"
        )

        st.info(
            f"Confidence: {round(confidence_score * 100, 2)}%"
        )
