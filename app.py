import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Animal Detector")

st.title("🐾 Animal Detector")

# Load model
model = tf.keras.models.load_model(
    "keras_model.h5",
    compile=False
)

# Load labels
class_names = open("labels.txt", "r").readlines()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Animal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    if st.button("Detect Animal"):

        # Resize
        image = image.resize((224, 224))

        # Convert to array
        image_array = np.asarray(image)

        # Normalize
        normalized_image_array = (
            image_array.astype(np.float32) / 127.5
        ) - 1

        # Prepare input
        data = np.ndarray(
            shape=(1, 224, 224, 3),
            dtype=np.float32
        )

        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)

        index = np.argmax(prediction)

        class_name = class_names[index]

        confidence_score = prediction[0][index]

        st.success(
            f"Animal: {class_name[2:].strip()}"
        )

        st.info(
            f"Confidence: {round(confidence_score * 100, 2)}%"
        )
