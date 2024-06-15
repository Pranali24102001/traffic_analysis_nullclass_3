#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the models
@st.cache(allow_output_mutation=True)
def load_models():
    car_color_model = tf.keras.models.load_model('car_color_model.h5')
    person_model = tf.keras.models.load_model('person_model.h5')
    traffic_model = tf.keras.models.load_model('traffic_model.h5')
    return car_color_model, person_model, traffic_model

car_color_model, person_model, traffic_model = load_models()

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize the image
    img = img.reshape(1, 224, 224, 3)  # Reshape for model input
    return img

# Function to make predictions
def predict(image, model):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return prediction

# Streamlit app
st.title("Traffic Analysis Model")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make predictions using each model
    st.subheader("Car Color Prediction:")
    car_color_prediction = predict(image, car_color_model)
    st.write(car_color_prediction)

    st.subheader("Person Detection Prediction:")
    person_prediction = predict(image, person_model)
    st.write(person_prediction)

    st.subheader("Traffic Vehicle Prediction:")
    traffic_prediction = predict(image, traffic_model)
    st.write(traffic_prediction)

