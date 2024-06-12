import streamlit as st
import tensorflow as tf 
from tensorflow import keras
import numpy as np
#import requests,os

import os
import requests
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

st.title("Welcome to Image Classifier")
st.header("Drop an image and we will tell whether its AI generated or not")
def download_model():
  
    os.path.exists('/workspaces/IMAGE_CLASSIFIER/ai_imageclassifier.h5')
    print(f"Model already exists at {'/workspaces/IMAGE_CLASSIFIER/ai_imageclassifier.h5'}.")
    return

def load_model():
    download_model()
    model=keras.models.load_model("/workspaces/IMAGE_CLASSIFIER/ai_imageclassifier.h5")
    return model


model = tf.keras.Sequential([
    # Previous layers...
    tf.keras.layers.Flatten(),  # Flatten the output to (1, 13456)
    tf.keras.layers.Dense(2048, activation='relu'),  # Dense layer with 2048 units
    tf.keras.layers.Dense(4096, activation='relu'),  # Dense layer with 4096 units
    # More layers...
])



model=load_model()
def predict(file):
    img = keras.utils.load_img(file, target_size=(256, 256))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.image.resize(img_array, (256, 256))  # Resize the image to match the model's input shape
    img_array /= 255.0  # Normalize the pixel values to be between 0 and 1
    pred = model.predict(img_array)
    if np.argmax(tf.nn.softmax(pred[0])) == 0:
        st.write("AI Generated")
    else:
        st.write("Real")



    
st.title("AI Image Classifier")
with st.spinner("Loading"):
    file=st.file_uploader("Upload Image",accept_multiple_files=False,type=['jpg','png','jpeg'])
if file is not None:
    content=file.getvalue()
    st.image(content)
    with st.spinner("Predicting"):
        predict(file)

