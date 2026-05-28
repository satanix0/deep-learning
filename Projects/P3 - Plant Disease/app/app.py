"""Streamlit app for classifying plant disease images with a saved model."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from keras_preprocessing import image
import numpy as np
import streamlit as st
import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "plant_disease_model.h5"
CLASS_INDEX_PATH = APP_DIR / "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with CLASS_INDEX_PATH.open(encoding="utf-8") as class_file:
    class_indices = json.load(class_file)


def load_and_preprocess(image_path, target_size=(256, 256)):
    """Load an image file and scale pixels into the model input range."""
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


def predict_image(model, image_path, class_indices):
    """Predict the class label for a preprocessed plant image."""
    preprocessed_image = load_and_preprocess(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    return class_indices[str(predicted_class_idx)]


def main():
    """Render the upload form and prediction result."""
    st.title("Plant Disease Prediction")

    uploaded_img = st.file_uploader(
        "Upload an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_img is None:
        st.warning("Please upload an image to classify.")
        return

    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.load_img(uploaded_img, target_size=(256, 256))
        st.image(resized_img, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button(label="Classify"):
            with st.spinner("Predicting..."):
                suffix = Path(uploaded_img.name).suffix
                with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_img.getbuffer())
                    temp_path = temp_file.name

                try:
                    predicted_class = predict_image(model, temp_path, class_indices)
                    st.success(f"Predicted Class: {predicted_class}")
                finally:
                    os.remove(temp_path)


if __name__ == "__main__":
    main()
