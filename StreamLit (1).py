import streamlit as st
import tensorflow as tf 
from tensorflow import keras
import numpy as np
#import requests,os

import os
import requests
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D

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


model = Sequential()

model.add(Conv2D(16, (4, 4), 1, activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (4, 4), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (4, 4), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



model=load_model()



def predict(file):
    img = keras.utils.load_img(file, target_size=(150, 150))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.image.resize(img_array, (150, 150))  # Resize the image to match the model's input shape
    img_array /= 255.0  # Normalize the pixel values to be between 0 and 1
    pred = model.predict(img_array)
    if(pred[0] > 0.5):
        st.write("AI generated")
    else:
        st.write("real image")
    



    
st.title("AI Image Classifier")
with st.spinner("Loading"):
    file=st.file_uploader("Upload Image",accept_multiple_files=False,type=['jpg','png','jpeg'])
if file is not None:
    content=file.getvalue()
    st.image(content)
    with st.spinner("Predicting"):
        predict(file)
