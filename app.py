
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detector")
st.write("Upload a plant leaf image to predict its disease.")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease.keras")

model = load_model()

# Disease categories - must match your training class order
class_names = [
    "Corn-Common_rust",
    "Potato-Early_blight",
    "Tomato-Bacterial_spot"
]

# Upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease", use_container_width=True):
        # Preprocess image - must match training preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))

        # Display result
        if len(class_names) != predictions.shape[-1]:
            st.error(
                f"Model outputs {predictions.shape[-1]} classes, "
                f"but {len(class_names)} class names were provided."
            )
        else:
            st.success("Prediction completed!")
            st.subheader("Predicted Disease")
            st.markdown(f"### 🌱 {class_names[predicted_index]}")

            confidence = float(predictions[0][predicted_index]) * 100
            st.write(f"Prediction score: **{confidence:.2f}%**")
