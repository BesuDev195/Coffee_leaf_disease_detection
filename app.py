"""
app.py
Robusta Coffee Leaf Disease Detection & Diagnosis AI System
Interactive Local Streamlit Web Application
"""

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Robusta Coffee Leaf Disease AI",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished agronomic card UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #40916c;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e9ecef;
    }
    .badge-healthy {
        background-color: #d8f3dc;
        color: #1b4332;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-rust {
        background-color: #ffccd5;
        color: #800f2f;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-mites {
        background-color: #ffe5d9;
        color: #9d0208;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 700;
        display: inline-block;
    }
    .treatment-box {
        background-color: #f1f8f4;
        border-left: 4px solid #2d6a4f;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Constants & Agronomic Treatment Knowledge Base
# ==============================================================================
MODEL_FILENAME = "coffee_disease_model.keras"
CLASS_INDICES_FILENAME = "class_indices.json"
TARGET_IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD_WARNING = 0.60  # 60% confidence warning

# Default class order if class_indices.json is absent
DEFAULT_CLASS_INDICES = {"healthy": 0, "mites": 1, "rust": 2}

CLASS_METADATA = {
    "healthy": {
        "display_name": "Healthy Leaf",
        "badge": "🟢 Healthy",
        "badge_class": "badge-healthy",
        "scientific_name": "Coffea canephora (Robusta)",
        "summary": "Leaf displays normal chlorosis-free tissue, intact epidermis, and vigorous physiological state.",
        "treatment": {
            "immediate": "No chemical application required. Continue regular scouting routine.",
            "cultural": [
                "Maintain balanced N-P-K soil fertilization based on seasonal soil tests.",
                "Execute structural pruning (desuckering) to maintain optimal solar interception and airflow.",
                "Ensure weed-free basins around the tree canopy to prevent nutrient competition."
            ],
            "monitoring": "Bi-weekly visual inspection during peak rainy season."
        }
    },
    "rust": {
        "display_name": "Coffee Leaf Rust",
        "badge": "🔴 Rust",
        "badge_class": "badge-rust",
        "scientific_name": "Hemileia vastatrix",
        "summary": "Fungal infection causing powdery orange-yellow lesions on leaf undersides, inducing severe premature defoliation.",
        "treatment": {
            "immediate": "Apply protective copper-based fungicide (e.g., Copper Oxychloride 50% WP at 2-3 g/L water) or systemic triazole fungicide in severe cases.",
            "cultural": [
                "Prune heavily infected, fallen, or dead branches and burn/bury them away from the plantation.",
                "Thin shade trees to reduce prolonged relative humidity (>80%) and promote leaf drying.",
                "Improve soil potassium and organic matter to strengthen leaf cuticle integrity."
            ],
            "monitoring": "Scout lower canopy every 7 days; spray protectants prior to onset of major rains."
        }
    },
    "mites": {
        "display_name": "Red Spider Mite Infestation",
        "badge": "🟠 Mites",
        "badge_class": "badge-mites",
        "scientific_name": "Oligonychus coffeae",
        "summary": "Acarid pest sucking sap from the upper leaf surface, causing bronzing, yellow speckling, and leaf drop.",
        "treatment": {
            "immediate": "Apply cold-pressed Horticultural Neem Oil (0.5%–1.0%) or registered selective miticide (e.g., Abamectin / Spiromesifen) targeting upper leaf surface.",
            "cultural": [
                "Dampen roadside coffee borders or plant barrier windbreaks to suppress dusty conditions favoring mite proliferation.",
                "Avoid broad-spectrum synthetic pyrethroids that eliminate predatory phytoseiid mites.",
                "Provide supplemental irrigation or overhead misting during prolonged dry, hot spells."
            ],
            "monitoring": "Check upper leaf surfaces using a 10x hand lens during warm, dry periods."
        }
    }
}

# ==============================================================================
# Model & Class Indices Loading (with Demo Mode fallback)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_disease_model():
    """
    Attempts to load the trained Keras model.
    Returns (model, is_demo_mode, error_message).
    """
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / MODEL_FILENAME

    if not model_path.exists():
        return None, True, f"Model file '{MODEL_FILENAME}' was not found in {current_dir}."

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
        return model, False, None
    except Exception as e:
        return None, True, f"Error loading '{MODEL_FILENAME}': {str(e)}"


def load_class_indices():
    """
    Loads class_indices.json mapping or defaults to DEFAULT_CLASS_INDICES.
    Returns (class_to_idx, idx_to_class).
    """
    current_dir = Path(__file__).resolve().parent
    mapping_path = current_dir / CLASS_INDICES_FILENAME

    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                raw_mapping = json.load(f)
                class_to_idx = {k: int(v) for k, v in raw_mapping.items()}
        except Exception:
            class_to_idx = DEFAULT_CLASS_INDICES
    else:
        class_to_idx = DEFAULT_CLASS_INDICES

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


# ==============================================================================
# Image Preprocessing & Inference
# ==============================================================================
def preprocess_leaf_image(image: Image.Image):
    """
    Prepares a PIL Image for inference matching MobileNetV2 requirements:
    1. Converts image to RGB (3 channels).
    2. Resizes to 224x224 using high-quality Bilinear filter.
    3. Normalizes pixel values: (img / 127.5) - 1.0 (range [-1, 1]).
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize
    image_resized = image.resize(TARGET_IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(image_resized, dtype=np.float32)
    
    # MobileNetV2 preprocessing formula
    img_preprocessed = (img_array / 127.5) - 1.0
    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch


def predict_disease(image: Image.Image, model, is_demo_mode: bool, idx_to_class: dict):
    """
    Runs model inference or generates a deterministic simulated prediction in Demo Mode.
    Returns: (predicted_class, confidence_float, all_probabilities_dict)
    """
    num_classes = len(idx_to_class)

    if not is_demo_mode and model is not None:
        batch = preprocess_leaf_image(image)
        raw_probs = model.predict(batch, verbose=0)[0]
        # Softmax normalization guard
        probabilities = np.exp(raw_probs) / np.sum(np.exp(raw_probs)) if np.max(raw_probs) > 1.0 else raw_probs
    else:
        # Realistic mock predictions for UI testing
        # Generate pseudo-random deterministic distribution based on image byte signature
        img_bytes = image.tobytes()
        seed_val = sum(img_bytes[:100]) if img_bytes else 42
        rng = np.random.default_rng(seed_val)
        simulated = rng.dirichlet(np.array([2.0, 2.0, 2.0]))
        # Boost one class to simulate a decisive prediction
        top_idx = int(rng.integers(0, num_classes))
        simulated[top_idx] += 1.8
        probabilities = simulated / np.sum(simulated)

    best_idx = int(np.argmax(probabilities))
    predicted_class = idx_to_class.get(best_idx, "healthy")
    confidence = float(probabilities[best_idx])

    probs_dict = {
        idx_to_class.get(i, f"class_{i}"): float(probabilities[i])
        for i in range(num_classes)
    }
    return predicted_class, confidence, probs_dict


# ==============================================================================
# UI Component: Sidebar
# ==============================================================================
def render_sidebar(is_demo: bool, load_error: str):
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.title("🌿 Robusta AI Diagnostics")
        st.markdown(
            "Precision agronomy assistant for **Robusta (*Coffea canephora*)** growers, agronomists, and researchers."
        )

        st.divider()
        st.subheader("📡 System Status")
        if is_demo:
            st.warning("⚠️ **DEMO MODE ACTIVE**")
            st.caption(
                "Trained weights (`coffee_disease_model.keras`) not found locally. "
                "The app is generating simulated predictions for UI preview."
            )
            if load_error:
                with st.expander("Technical details"):
                    st.code(load_error, language="text")
        else:
            st.success("🟢 **LIVE INFERENCE ACTIVE**")
            st.caption("MobileNetV2 neural network weights loaded into memory.")

        st.divider()
        st.subheader("📖 Disease Field Guide")
        with st.expander("🟢 Healthy"):
            st.caption("Even emerald coloration, no rust pustules or bronze stippling.")
        with st.expander("🔴 Coffee Leaf Rust (*Hemileia vastatrix*)"):
            st.caption("Yellow-orange powdery fungal pustules starting on the lower leaf surface.")
        with st.expander("🟠 Spider Mites (*Oligonychus coffeae*)"):
            st.caption("Bronze/yellow speckled discoloration caused by sap-feeding mites.")

        st.divider()
        st.caption("Robusta Leaf AI • v1.0.0 • MobileNetV2 Backbone")


# ==============================================================================
# Main Application Flow
# ==============================================================================
def main():
    # Load model and mappings
    model, is_demo, load_error = load_disease_model()
    class_to_idx, idx_to_class = load_class_indices()

    render_sidebar(is_demo, load_error)

    # Top Header
    st.markdown('<div class="main-header">☕ Robusta Coffee Leaf Disease Diagnosis AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload high-resolution Robusta coffee leaf photos for real-time pathology classification and expert agronomic treatment plans.</div>',
        unsafe_allow_html=True,
    )

    if is_demo:
        st.info(
            "💡 **Running in Demo Mode**: To enable real deep learning inference, train the model using "
            "`rocole_coffee_disease_cnn.ipynb` and place `coffee_disease_model.keras` and `class_indices.json` in this app's folder.",
            icon="ℹ️"
        )

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Leaf Photos (Single or Multiple)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supports JPG, JPEG, and PNG images. For best accuracy, capture well-lit leaves filling the camera frame."
    )

    if not uploaded_files:
        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown("### 1. Capture")
            st.write("Take a sharp photo of the top or bottom of a Robusta leaf showing symptoms or healthy tissue.")
        with col_info2:
            st.markdown("### 2. Diagnose")
            st.write("MobileNetV2 analyzes fungal sporulation, chlorosis, and mite damage patterns.")
        with col_info3:
            st.markdown("### 3. Treat")
            st.write("Receive actionable, field-tested integrated pest management (IPM) agronomic prescriptions.")
        return

    st.markdown(f"### 📋 Diagnostic Results ({len(uploaded_files)} image{'s' if len(uploaded_files) > 1 else ''})")

    # Process each uploaded file
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        col_img, col_diag = st.columns([1, 1.4], gap="medium")

        # 1. Image Preview & Validation
        try:
            image = Image.open(uploaded_file)
            image.verify()  # Integrity check
            image = Image.open(uploaded_file)  # Re-open after verify
        except Exception as e:
            with col_img:
                st.error(f"❌ Corrupt or invalid image: {uploaded_file.name}")
                st.caption(f"Reason: {str(e)}")
            continue

        with col_img:
            st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=True)
            st.caption(f"Original Resolution: {image.size[0]} × {image.size[1]} px | Mode: {image.mode}")

        # 2. Run Inference
        with col_diag:
            with st.spinner("Analyzing leaf pathology..."):
                pred_class, confidence, probs = predict_disease(image, model, is_demo, idx_to_class)

            meta = CLASS_METADATA.get(pred_class, CLASS_METADATA["healthy"])

            # Classification Badge & Confidence
            badge_text = meta["badge"]
            badge_class = meta["badge_class"]
            conf_percent = confidence * 100.0

            st.markdown(
                f'<span class="{badge_class}" style="font-size: 1.25rem;">{badge_text}</span>'
                f'<span style="font-size: 1.25rem; font-weight: 700; margin-left: 15px;">Confidence: {conf_percent:.1f}%</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Pathogen / Agent:** *{meta['scientific_name']}*")
            st.write(meta["summary"])

            # Low Confidence Alert
            if confidence < CONFIDENCE_THRESHOLD_WARNING:
                st.warning(
                    f"⚠️ **Low Confidence Warning ({conf_percent:.1f}%)**: The prediction is below the 60% confidence threshold. "
                    "Possible causes: blurry focus, unusual lighting, or co-infection. "
                    "Consider capturing a clearer close-up or consulting an agricultural extension officer.",
                    icon="⚠️"
                )

            # Class Probability Bars
            st.markdown("##### Class Probabilities:")
            for c_name in ["healthy", "rust", "mites"]:
                c_prob = probs.get(c_name, 0.0)
                st.write(f"• **{c_name.capitalize()}**: `{c_prob * 100:.1f}%`")
                st.progress(float(min(max(c_prob, 0.0), 1.0)))

            # Agronomic Treatment Plan
            st.markdown("##### 🌾 Agronomic Prescription & Action Plan:")
            treatment = meta["treatment"]
            with st.container():
                st.markdown(
                    f"""
                    <div class="treatment-box">
                        <b>⚡ Immediate Action:</b><br>{treatment['immediate']}<br><br>
                        <b>🌱 Cultural & Canopy Management:</b>
                        <ul>
                            {''.join(f'<li>{item}</li>' for item in treatment['cultural'])}
                        </ul>
                        <b>🔍 Monitoring Protocol:</b><br>{treatment['monitoring']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
