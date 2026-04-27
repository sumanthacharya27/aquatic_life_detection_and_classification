# 🐠 Aquatic Life Classification System

## Overview
A CNN-based image classification app that identifies **8 aquatic species** from uploaded images.
Built with MobileNetV2 transfer learning and deployed as a **Streamlit web app**.

## Supported Species
🦀 Crab · 🐬 Dolphin · 🐙 Octopus · 🌊 Seahorse · 🦭 Seal · 🐢 Sea Turtle · 🦈 Shark · ⭐ Starfish

## Features
- Multi-class classification (8 species)
- Transfer learning with MobileNetV2
- Confidence scoring with low-confidence warnings
- Probability bar chart across all classes
- Species info panel (category, habitat, diet, lifespan)
- Robust error handling

## Tech Stack
| Component | Technology |
|-----------|-----------|
| ML Framework | TensorFlow 2.15 / Keras |
| Base Model | MobileNetV2 (Transfer Learning) |
| Image Processing | Pillow, NumPy |
| Web UI | Streamlit |
| Training Environment | Google Colab |

## Project Structure
```
Aquatic_classification_and detection/
├── app/
│   └── app.py                  # Streamlit web app
├── model/
│   ├── aquatic_model_v1.h5     # Keras H5 format
│   ├── aquatic_model_v1.keras  # Keras native format
│   └── aquatic_savedmodel/     # TF SavedModel (used by app)
├── notebook/
│   └── aquaticLife_Detect_classif.ipynb  # Training notebook
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

## Model Training
See `notebook/aquaticLife_Detect_classif.ipynb` for the full training pipeline (Google Colab).

## Future Work
- Video frame-by-frame classification
- Object detection with bounding boxes (YOLO)
- Multi-image batch upload support
- Model retraining pipeline with new species