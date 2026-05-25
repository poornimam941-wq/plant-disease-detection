
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Disease information
disease_info = {

    "Peach___Bacterial_spot": {
        "cause": "Bacterial infection affecting peach leaves.",
        "symptoms": [
            "Dark leaf spots",
            "Damaged leaf tissue",
            "Yellowing around spots"
        ],
        "treatment": [
            "Remove infected leaves",
            "Use copper-based bactericide",
            "Avoid overhead watering"
        ],
        "prevention": [
            "Maintain airflow",
            "Avoid overwatering",
            "Inspect leaves regularly"
        ]
    },

    "Tomato___Early_blight": {
        "cause": "Fungal infection caused by Alternaria solani.",
        "symptoms": [
            "Dark brown spots",
            "Yellow edges",
            "Leaf drying"
        ],
        "treatment": [
            "Apply fungicide",
            "Remove infected leaves",
            "Avoid wet foliage"
        ],
        "prevention": [
            "Crop rotation",
            "Healthy soil",
            "Proper drainage"
        ]
    }
}

# Class names
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Title
st.title("🌱 Smart Plant Disease Detection System")

st.write(
    "Upload a plant leaf image and let AI detect disease."
)

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    image_np = np.array(image)

    image_resized = cv2.resize(image_np, (128,128))

    image_input = image_resized / 255.0
    image_input = np.expand_dims(image_input, axis=0)

    prediction = model.predict(image_input)[0]

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    disease_name = class_names[predicted_index]

    # Severity level
    if confidence >= 90:
        severity = "🔴 High Confidence Detection"
    elif confidence >= 70:
        severity = "🟠 Moderate Confidence Detection"
    else:
        severity = "🟡 Low Confidence Detection"

    # Results section
    st.success(f"🌱 Disease: {disease_name}")

    st.subheader("📊 Confidence Score")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}%")

    st.warning(f"⚠️ {severity}")

    # Smart recommendation
    if disease_name in disease_info:

        st.subheader("🦠 Cause")
        st.write(disease_info[disease_name]["cause"])

        st.subheader("🔍 Symptoms")
        for symptom in disease_info[disease_name]["symptoms"]:
            st.write("•", symptom)

        st.subheader("💊 Treatment")
        for treatment in disease_info[disease_name]["treatment"]:
            st.write("•", treatment)

        st.subheader("🛡️ Prevention")
        for prevention in disease_info[disease_name]["prevention"]:
            st.write("•", prevention)

    else:
        st.info("Detailed recommendation coming soon.")