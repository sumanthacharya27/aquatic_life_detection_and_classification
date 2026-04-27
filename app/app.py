import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.saved_model.load("model/aquatic_savedmodel")
infer = model.signatures["serving_default"]

# Class names (MAKE SURE ORDER MATCHES TRAINING)
class_names = ['crab','dolphin','octopus','seahorse', 'seal', 'seaturtle','shark', 'starfish']

# Info dictionary
animal_info = {
    "dolphin": {"category": "Mammal", "habitat": "Ocean", "diet": "Carnivore", "lifespan": "40–60 years"},
    "shark": {"category": "Fish", "habitat": "Ocean", "diet": "Carnivore", "lifespan": "20–30 years"},
    "octopus": {"category": "Mollusk", "habitat": "Ocean", "diet": "Carnivore", "lifespan": "1–3 years"},
    "crab": {"category": "Crustacean", "habitat": "Sea/shore", "diet": "Omnivore", "lifespan": "3–5 years"},
    "starfish": {"category": "Echinoderm", "habitat": "Ocean", "diet": "Carnivore", "lifespan": "5–35 years"},
    "seahorse": {"category": "Fish", "habitat": "Ocean", "diet": "Carnivore", "lifespan": "1–5 years"},
    "seal": {"category": "Mammal", "habitat": "Ocean", "diet": "Carnivore", "lifespan": "25–30 years"},
    "seaturtle": {"category": "Reptile", "habitat": "Ocean", "diet": "Omnivore", "lifespan": "50–100 years"}
}

st.title(" Aquatic Life classification System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224)).convert("RGB")  # convert ensures no RGBA issues
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction ← CHANGED BLOCK
    img_tensor = tf.constant(img_array, dtype=tf.float32)
    output = infer(img_tensor)
    prediction = list(output.values())[0].numpy()

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

    # Info panel
    if predicted_class in animal_info:
        info = animal_info[predicted_class]
        st.markdown("### Details:")
        st.write(f"Category: {info['category']}")
        st.write(f"Habitat: {info['habitat']}")
        st.write(f"Diet: {info['diet']}")
        st.write(f"Lifespan: {info['lifespan']}")