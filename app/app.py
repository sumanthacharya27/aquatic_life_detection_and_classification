import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aquatic Life Detection System",
    page_icon="🌊",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "model" / "aquatic_model_v2.keras"
YOLO_PATH  = Path(__file__).parent.parent / "model" / "aquatic_yolo_best.pt"
CONFIDENCE_THRESHOLD = 0.60

CLASS_NAMES = [
    "crab", "dolphin", "jellyfish", "octopus", "seahorse",
    "seal", "seaturtle", "shark", "squid", "starfish",
]

YOLO_CLASS_NAMES = ['crab', 'dolphin', 'octopus', 'shark', 'starfish', 'turtle']

ANIMAL_INFO = {
    "crab":      {"category": "Crustacean", "habitat": "Sea/Shore", "diet": "Omnivore",  "lifespan": "3–5 years"},
    "dolphin":   {"category": "Mammal",     "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "40–60 years"},
    "jellyfish": {"category": "Cnidaria",   "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "1 year"},
    "octopus":   {"category": "Mollusk",    "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "1–3 years"},
    "seahorse":  {"category": "Fish",       "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "1–5 years"},
    "seal":      {"category": "Mammal",     "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "25–30 years"},
    "seaturtle": {"category": "Reptile",    "habitat": "Ocean",     "diet": "Omnivore",  "lifespan": "50–100 years"},
    "shark":     {"category": "Fish",       "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "20–30 years"},
    "squid":     {"category": "Mollusk",    "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "1–5 years"},
    "starfish":  {"category": "Echinoderm", "habitat": "Ocean",     "diet": "Carnivore", "lifespan": "5–35 years"},
    "turtle":    {"category": "Reptile",    "habitat": "Ocean",     "diet": "Omnivore",  "lifespan": "50–100 years"},
}

SPECIES_EMOJI = {
    "crab": "🦀", "dolphin": "🐬", "jellyfish": "🪼", "octopus": "🐙",
    "seahorse": "🌊", "seal": "🦭", "seaturtle": "🐢", "shark": "🦈",
    "squid": "🦑", "starfish": "⭐", "turtle": "🐢",
}

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    try:
        return tf.keras.models.load_model(str(MODEL_PATH))
    except Exception as e:
        st.error(f"❌ Failed to load classifier: {e}")
        st.stop()

@st.cache_resource
def load_yolo():
    try:
        return YOLO(str(YOLO_PATH))
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        st.stop()

classifier = load_classifier()
yolo_model = load_yolo()

# ── Grad-CAM helpers ──────────────────────────────────────────────────────────
def generate_gradcam(model, img_array, pred_index):
    try:
        base_model = model.get_layer("mobilenetv2_1.00_224")

        grad_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[
                base_model.get_layer("out_relu").output,
                base_model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, _ = grad_model(img_array)
            x           = model.get_layer("global_average_pooling2d_1")(conv_outputs)
            x           = model.get_layer("dense_2")(x)
            x           = model.get_layer("dropout_1")(x, training=False)
            predictions = model.get_layer("dense_3")(x)
            loss        = predictions[:, pred_index]

        grads        = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        return None, str(e)

def overlay_gradcam(original_img, heatmap):
    img             = np.array(original_img.resize((224, 224)).convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed    = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(superimposed)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌊 Aquatic Life Detection System")
st.caption("Classify images · Detect animals in videos · Explainable AI heatmaps")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🖼️ Image Classification",
    "🎥 Video Detection",
    "🧠 Explainable AI",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Upload an image to classify the aquatic species")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG / PNG / JPEG)",
        type=["jpg", "png", "jpeg"],
        key="image_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        try:
            img       = image.resize((224, 224)).convert("RGB")
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = classifier.predict(img_array, verbose=0)[0]
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
            st.stop()

        predicted_idx   = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(predictions[predicted_idx])
        emoji           = SPECIES_EMOJI.get(predicted_class, "🐟")

        with col2:
            st.markdown("### 🔍 Prediction")
            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ Low confidence ({confidence:.0%}). "
                    "Try a clearer image of an aquatic animal."
                )
            else:
                st.success(f"{emoji} **{predicted_class.capitalize()}**")

            st.metric(label="Confidence", value=f"{confidence:.1%}")

            if predicted_class in ANIMAL_INFO:
                info = ANIMAL_INFO[predicted_class]
                st.markdown("#### 📋 Species Info")
                for k, v in {
                    "Category": info["category"],
                    "Habitat":  info["habitat"],
                    "Diet":     info["diet"],
                    "Lifespan": info["lifespan"],
                }.items():
                    st.markdown(f"- **{k}:** {v}")

        st.divider()
        st.markdown("### 📊 Confidence Across All Classes")
        chart_data = dict(sorted(
            {n.capitalize(): float(predictions[i]) for i, n in enumerate(CLASS_NAMES)}.items(),
            key=lambda x: x[1], reverse=True
        ))
        df = pd.DataFrame(
            {"Species": list(chart_data.keys()), "Confidence": list(chart_data.values())}
        ).set_index("Species")
        st.bar_chart(df, height=260)

    else:
        st.info("👆 Upload an image above to get started.")
        st.markdown(
            """
            **Supported species:**
            🦀 Crab · 🐬 Dolphin · 🪼 Jellyfish · 🐙 Octopus · 🌊 Seahorse
            · 🦭 Seal · 🐢 Sea Turtle · 🦈 Shark · 🦑 Squid · ⭐ Starfish
            """
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Upload a video to detect aquatic animals with bounding boxes")
    st.caption("Supported species: 🦀 Crab · 🐬 Dolphin · 🐙 Octopus · 🦈 Shark · ⭐ Starfish · 🐢 Turtle")

    uploaded_video = st.file_uploader(
        "Upload a video (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.20, 0.05)

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_video.read())
            input_path = tmp_in.name

        output_path = input_path.replace(".mp4", "_output.mp4")

        cap          = cv2.VideoCapture(input_path)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        progress_bar   = st.progress(0, text="Processing video...")
        frame_count    = 0
        detections_log = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model.track(frame, conf=conf_threshold, persist=True, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                label    = YOLO_CLASS_NAMES[cls_id] if cls_id < len(YOLO_CLASS_NAMES) else "unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 150), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf_val:.0%}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 150), 2
                )
                detections_log.append(label)

            out.write(frame)
            frame_count += 1

            if total_frames > 0:
                progress_bar.progress(
                    min(frame_count / total_frames, 1.0),
                    text=f"Processing frame {frame_count}/{total_frames}..."
                )

        cap.release()
        out.release()
        progress_bar.empty()
        st.success("✅ Video processed!")

        with open(output_path, "rb") as f:
            st.video(f.read())

        if detections_log:
            st.divider()
            st.markdown("### 📊 Detection Summary")
            counts     = Counter(detections_log)
            summary_df = pd.DataFrame(
                {"Species": list(counts.keys()), "Detections": list(counts.values())}
            ).sort_values("Detections", ascending=False).set_index("Species")
            st.bar_chart(summary_df, height=200)
        else:
            st.info("No aquatic animals detected. Try lowering the confidence threshold.")

    else:
        st.info("👆 Upload a video above to get started.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPLAINABLE AI (Grad-CAM)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🧠 Explainable AI — Grad-CAM Heatmap")
    st.markdown(
        "Grad-CAM shows **which parts of the image** the model focused on to make "
        "its prediction. Red/yellow = high attention. Blue = low attention. "
        "Ideally the animal's body should be highlighted, not the background."
    )
    st.divider()

    xai_file = st.file_uploader(
        "Upload an image to analyse (JPG / PNG / JPEG)",
        type=["jpg", "png", "jpeg"],
        key="xai_uploader"
    )

    if xai_file is not None:
        xai_image = Image.open(xai_file)

        try:
            img       = xai_image.resize((224, 224)).convert("RGB")
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = classifier.predict(img_array, verbose=0)[0]
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
            st.stop()

        predicted_idx   = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(predictions[predicted_idx])
        emoji           = SPECIES_EMOJI.get(predicted_class, "🐟")

        st.markdown(f"**Prediction:** {emoji} {predicted_class.capitalize()} — `{confidence:.1%}` confidence")
        st.divider()

        heatmap = generate_gradcam(classifier, img_array, predicted_idx)

        if heatmap is not None and not isinstance(heatmap, tuple):
            gradcam_img = overlay_gradcam(xai_image, heatmap)

            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    xai_image.resize((224, 224)),
                    caption="Original Image",
                    use_container_width=True
                )
            with c2:
                st.image(
                    gradcam_img,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )

            st.divider()
            st.markdown("#### 🔍 How to interpret this heatmap")
            st.markdown(
                """
                - **Red/yellow on the animal** → model is learning correctly
                - **Red/yellow on the background** → model may be using wrong features (e.g. water color, background texture)
                - **Diffuse heatmap spread everywhere** → model is uncertain
                
                If the heatmap consistently highlights backgrounds instead of animals, 
                the training data likely has a background bias and the model needs more diverse images.
                """
            )
        else:
            st.error(
                "❌ Grad-CAM failed. This usually means the layer name `Conv_1` "
                "doesn't match your model architecture. Check your model's layer names with: "
                "`[l.name for l in classifier.layers]`"
            )

    else:
        st.info("👆 Upload an image above to generate the Grad-CAM heatmap.")