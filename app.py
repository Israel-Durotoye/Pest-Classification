import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import gdown

url = "https://drive.google.com/uc?export=download&id=1fnQOvlcQ3hjs7SPJJrf8EnKiui_Ukthu"
output = "pest_classification_model.h5"
gdown.download(url, output, quiet=False)

# Load the pre-trained model
model = tf.keras.models.load_model("pest_classification_model.h5")

# Define class names
class_names = {0: 'ants', 1: 'bees', 2: 'beetle', 3: 'catterpillar', 4: 'earthworms', 5: 'earwig', 6: 'grasshopper', 7: 'moth', 8: 'slug', 9: 'snail', 10: 'wasp', 11: 'weevil'}

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of the image
def predict_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Function to show sample images from a directory
def show_sample_images(directory, class_name, num_samples=5):
    class_dir = os.path.join(directory, class_name)
    sample_files = [file for file in os.listdir(class_dir) if file.endswith(('jpg', 'png', 'jpeg'))]
    sample_files = random.sample(sample_files, min(num_samples, len(sample_files)))
    
    for file in sample_files:
        img_path = os.path.join(class_dir, file)
        img = Image.open(img_path)
        st.image(img, caption=f'{class_name} - {file}', use_column_width=True)

# Streamlit app
st.title("Pest Classification App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class = predict_image(image)
    st.write(f"Predicted Class: {predicted_class}")

# Show sample images of a selected class
st.write("")
st.write("Show sample images of a class")
selected_class = st.selectbox("Select a class", list(class_names.values()))
if selected_class:
    st.write(f"Sample images of {selected_class}:")
    show_sample_images('pestdata/train', selected_class)