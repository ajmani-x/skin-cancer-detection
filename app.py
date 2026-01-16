import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

IMG_SIZE = 128  
MODEL_PATH = "cnn_malignant_vs_benign_v2.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

st.title("🧬 Skin Cancer Classification")
st.write("Upload a skin lesion image to classify it as **Benign** or **Malignant**.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=700)


    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0


    prediction = model.predict(img_array)[0][0]

    st.subheader("🧪 Prediction Result")

    if prediction > 0.5:
        st.error("🔴 Malignant")
        st.write(f"Confidence: {prediction * 100:.2f}%")
    else:
        st.success("🟢 Benign")
        st.write(f"Confidence: {(1 - prediction) * 100:.2f}%")
