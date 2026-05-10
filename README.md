
# 🐠 Aquatic Life Classification & Detection System

## Overview
This project is a web-based system for both **image classification** and **object detection** of aquatic species. It uses:
- A MobileNetV2-based Keras classifier for species recognition
- A YOLOv8 model for object detection in images
- Grad-CAM for visualizing model attention
All features are accessible via a user-friendly **Streamlit web app**.

## Supported Species
🦀 Crab · 🐬 Dolphin · 🪼 Jellyfish · 🐙 Octopus · 🌊 Seahorse · 🦭 Seal · 🐢 Sea Turtle · 🦈 Shark · 🦑 Squid · ⭐ Starfish

## Features
- Multi-class classification (10 species)
- Object detection (YOLOv8) for 6+ species
- Grad-CAM heatmap visualization for classifier predictions
- Transfer learning with MobileNetV2
- Confidence scoring with low-confidence warnings
- Probability bar chart across all classes
- Species info panel (category, habitat, diet, lifespan)
- Robust error handling

## Tech Stack
| Component         | Technology                        |
|-------------------|-----------------------------------|
| ML Framework      | TensorFlow 2.15 / Keras           |
| Object Detection  | YOLOv8 (ultralytics)              |
| Base Model        | MobileNetV2 (Transfer Learning)   |
| Image Processing  | Pillow, OpenCV, NumPy             |
| Data Analysis     | pandas                            |
| Web UI            | Streamlit                         |
| Visualization     | matplotlib, Grad-CAM              |
| Training Env      | Google Colab                      |

## Project Structure
```
Aquatic_classification_and_detection/
├── app/
│   └── app.py                  # Streamlit web app (classification & detection)
├── model/
│   ├── aquatic_model_v1.h5     # Keras H5 format
│   ├── aquatic_model_v1.keras  # Keras native format
│   ├── aquatic_model_v2.keras  # Keras classifier (used by app)
│   ├── aquatic_yolo_best.pt    # YOLOv8 model (used by app)
│   └── aquatic_savedmodel/     # TF SavedModel (legacy)
├── notebook/
│   ├── aquaticLife_Detect_classif.ipynb   # Classification training notebook
│   └── aquaticLife_v2_detect.ipynb        # Detection/experiments notebook
├── test_media/                 # Place test images here
├── fix_model.py                # Keras 3 → 2 H5 compatibility patch
└── requirements.txt
```

## How to Run

```bash
# 1. Install dependencies (from project root)
pip install -r requirements.txt

# 2. Launch the Streamlit app (must run from project root)
streamlit run app/app.py
```

> ⚠️ **Important:** Always run from the **project root directory**, not from inside `app/`.

## Usage

### Image Classification
- Upload an image of an aquatic animal.
- The app predicts the species, shows confidence, and displays a Grad-CAM heatmap.

### Object Detection
- Upload an image with multiple aquatic animals.
- The app uses YOLOv8 to detect and label all supported species in the image.

## Model Training
See `notebook/aquaticLife_Detect_classif.ipynb` and `notebook/aquaticLife_v2_detect.ipynb` for the full training and detection pipelines (Google Colab).

## Future Work
- Video frame-by-frame classification
- Improved object detection with bounding boxes (YOLO)
- Real-time webcam support
- Multi-image batch upload support
- Model retraining pipeline with new species