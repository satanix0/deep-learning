import os
import json
from keras_preprocessing import image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.getcwd()
model_path = os.path.join(working_dir, 'model', 'plant_disease_model.h5')
model = tf.keras.models.load_model(model_path)

class_indices = json.load(
    open(os.path.join(working_dir, 'class_indices.json'), 'r'))


def load_and_preprocess(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize to [0, 1]
    return img


def predict_image(model, image_path, class_indices):
    preprocessed_image = load_and_preprocess(image_path)
    predictions = model.predict(preprocessed_image)
    predict_class_idx = np.argmax(predictions, axis=1)[0]
    predict_class = class_indices[str(predict_class_idx)]
    return predict_class


def main():
    st.title("Plant Disease Prediction")

    uploaded_img = st.file_uploader(
        "Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        col1, col2 = st.columns(2)  # Create two columns

        with col1:
            resized_img = image.load_img(uploaded_img, target_size=(256, 256))
            st.image(resized_img, caption="Uploaded Image",
                     use_column_width=True)

        with col2:
            if st.button(label="classify"):
                with st.spinner("Predicting..."):
                    img_path = os.path.join(
                        working_dir, 'temp', uploaded_img.name)
                    # Ensure the temp directory exists
                    with open(img_path, "wb") as f:
                        # Save the uploaded file to a temporary location
                        f.write(uploaded_img.getbuffer())

                    predicted_class = predict_image(
                        model, img_path, class_indices)
                    st.success(f"Predicted Class: {predicted_class}")

                    # Clean up the temporary file
                    os.remove(img_path)
    else:
        st.warning("Please upload an image to classify.")
