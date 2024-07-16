import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Set up the Streamlit app
st.set_page_config(page_title="Image Classification Model", layout="centered")

# Header and description
st.title('Image Classification Model')
st.write("Upload an image of a fruit or vegetable, and the model will classify it.")

# Load the model
model = load_model('./Image_classify.keras')

# List of categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
            'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
            'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
            'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 
            'turnip', 'watermelon']

img_height = 180
img_width = 180

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Display the image
    st.image(image_load, caption='Uploaded Image', use_column_width=True)

    # Display the prediction results
    st.write(f'**Predicted Category:** {data_cat[np.argmax(score)]}')
    st.write(f'**Confidence:** {round(np.max(score) * 100, 2)}%')

# Styling
st.markdown("""
    <style>
        .css-1d391kg, .css-18e3th9 {
            text-align: center;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 14px 20px;
            margin: 8px 0;
            border: none;
            cursor: pointer;
            width: 100%;
            font-size: 17px;
        }
    </style>
    """, unsafe_allow_html=True)
