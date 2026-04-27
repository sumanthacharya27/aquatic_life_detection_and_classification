import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aquatic Life Classifier",
    page_icon="🐠",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "model" / "aquatic_savedmodel"
CONFIDENCE_THRESHOLD = 0.50

CLASS_NAMES = [
    "crab", "dolphin", "octopus", "seahorse",
    "seal", "seaturtle", "shark", "starfish",
]

ANIMAL_INFO = {
    "dolphin":   {"category": "Mammal",      "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "40–60 years"},
    "shark":     {"category": "Fish",         "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "20–30 years"},
    "octopus":   {"category": "Mollusk",      "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "1–3 years"},
    "crab":      {"category": "Crustacean",   "habitat": "Sea/Shore", "diet": "Omnivore",  "lifespan": "3–5 years"},
    "starfish":  {"category": "Echinoderm",   "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "5–35 years"},
    "seahorse":  {"category": "Fish",         "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "1–5 years"},
    "seal":      {"category": "Mammal",       "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "25–30 years"},
    "seaturtle": {"category": "Reptile",      "habitat": "Ocean",     "diet": "Omnivore",  "lifespan": "50–100 years"},
}

SPECIES_EMOJI = {
    "dolphin": "🐬", "shark": "🦈", "octopus": "🐙",
    "crab": "🦀", "starfish": "⭐", "seahorse": "🌊",
    "seal": "🦭", "seaturtle": "🐢",
}

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.saved_model.load(str(MODEL_PATH))
        return model.signatures["serving_default"]
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

infer = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐠 Aquatic Life Classifier")
st.caption("Upload an image of an aquatic animal and the model will identify the species.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG / JPEG)",
    type=["jpg", "png", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        img = image.resize((224, 224)).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        img_tensor = tf.constant(img_array, dtype=tf.float32)
        output = infer(img_tensor)
        predictions = list(output.values())[0].numpy()[0]   # shape: (8,)

    except Exception as e:
        st.error(f"❌ Inference failed: {e}")
        st.stop()

    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])
    emoji = SPECIES_EMOJI.get(predicted_class, "🐟")

    with col2:
        st.markdown("### 🔍 Prediction")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ Low confidence ({confidence:.0%}). The model is uncertain — "
                "try a clearer image of an aquatic animal."
            )
        else:
            st.success(f"{emoji} **{predicted_class.capitalize()}**")

        # Confidence meter
        st.metric(label="Confidence", value=f"{confidence:.1%}")

        # ── Info panel ────────────────────────────────────────────────────────
        if predicted_class in ANIMAL_INFO:
            info = ANIMAL_INFO[predicted_class]
            st.markdown("#### 📋 Species Info")
            info_table = {
                "Category": info["category"],
                "Habitat":  info["habitat"],
                "Diet":     info["diet"],
                "Lifespan": info["lifespan"],
            }
            for k, v in info_table.items():
                st.markdown(f"- **{k}:** {v}")

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Confidence Across All Classes")

    chart_data = {
        name.capitalize(): float(predictions[i])
        for i, name in enumerate(CLASS_NAMES)
    }
    # Sort descending
    chart_data = dict(sorted(chart_data.items(), key=lambda x: x[1], reverse=True))

    import pandas as pd
    df = pd.DataFrame(
        {"Species": list(chart_data.keys()), "Confidence": list(chart_data.values())}
    ).set_index("Species")

    st.bar_chart(df, height=260)

else:
    st.info("👆 Upload an image above to get started.")
    st.markdown(
        """
        **Supported species:**
        🦀 Crab · 🐬 Dolphin · 🐙 Octopus · 🌊 Seahorse
        · 🦭 Seal · 🐢 Sea Turtle · 🦈 Shark · ⭐ Starfish
        """
    )