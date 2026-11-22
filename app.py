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

# Custom CSS for polished, high-contrast agronomic card UI (Dark & Light Mode safe)
st.markdown(
    """
    <style>
    /* Main Layout Aesthetics */
    .hero-container {
        background: linear-gradient(135deg, #0f392b 0%, #1b5e20 50%, #2e7d32 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #ffffff !important;
        margin-bottom: 24px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c8e6c9 !important;
        line-height: 1.5;
        max-width: 900px;
        margin-bottom: 0px;
    }
    
    /* Result Diagnostic Card */
    .diag-container {
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    }
    
    /* Class Badges */
    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.2px;
    }
    .badge-healthy {
        background-color: #d1fae5;
        color: #065f46;
        border: 1.5px solid #10b981;
    }
    .badge-rust {
        background-color: #ffe4e6;
        color: #9f1239;
        border: 1.5px solid #f43f5e;
    }
    .badge-mites {
        background-color: #fef3c7;
        color: #92400e;
        border: 1.5px solid #f59e0b;
    }
    
    /* Custom Prescription Action Plan Cards */
    .rx-card {
        border-radius: 14px;
        padding: 22px 24px;
        margin-top: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        transition: transform 0.2s ease;
    }
    .rx-card-healthy {
        background-color: #f0fdf4 !important;
        border: 2px solid #34d399 !important;
    }
    .rx-card-rust {
        background-color: #fff1f2 !important;
        border: 2px solid #fb7185 !important;
    }
    .rx-card-mites {
        background-color: #fffbeb !important;
        border: 2px solid #fbbf24 !important;
    }
    
    .rx-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 16px;
        padding-bottom: 10px;
    }
    .rx-header-healthy {
        color: #064e3b !important;
        border-bottom: 2px dashed #a7f3d0;
    }
    .rx-header-rust {
        color: #881337 !important;
        border-bottom: 2px dashed #fecdd3;
    }
    .rx-header-mites {
        color: #78350f !important;
        border-bottom: 2px dashed #fde68a;
    }
    
    /* Section Boxes */
    .rx-section {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .rx-section-label {
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 6px;
    }
    
    /* Healthy Section Colors */
    .rx-box-healthy-action {
        background-color: #dcfce7 !important;
        border-left: 5px solid #16a34a !important;
        color: #14532d !important;
    }
    .rx-box-healthy-action .rx-section-label {
        color: #15803d !important;
    }
    
    .rx-box-healthy-culture {
        background-color: #ffffff !important;
        border-left: 5px solid #10b981 !important;
        color: #1f2937 !important;
        border: 1px solid #d1fae5;
    }
    .rx-box-healthy-culture .rx-section-label {
        color: #047857 !important;
    }
    
    .rx-box-healthy-monitor {
        background-color: #e0f2fe !important;
        border-left: 5px solid #0284c7 !important;
        color: #0c4a6e !important;
    }
    .rx-box-healthy-monitor .rx-section-label {
        color: #0369a1 !important;
    }
    
    /* Rust Section Colors */
    .rx-box-rust-action {
        background-color: #ffe4e6 !important;
        border-left: 5px solid #e11d48 !important;
        color: #881337 !important;
    }
    .rx-box-rust-action .rx-section-label {
        color: #be123c !important;
    }
    
    .rx-box-rust-culture {
        background-color: #ffffff !important;
        border-left: 5px solid #f43f5e !important;
        color: #1f2937 !important;
        border: 1px solid #ffe4e6;
    }
    .rx-box-rust-culture .rx-section-label {
        color: #be123c !important;
    }
    
    .rx-box-rust-monitor {
        background-color: #fef2f2 !important;
        border-left: 5px solid #b91c1c !important;
        color: #7f1d1d !important;
    }
    .rx-box-rust-monitor .rx-section-label {
        color: #991b1b !important;
    }
    
    /* Mites Section Colors */
    .rx-box-mites-action {
        background-color: #fef3c7 !important;
        border-left: 5px solid #d97706 !important;
        color: #78350f !important;
    }
    .rx-box-mites-action .rx-section-label {
        color: #b45309 !important;
    }
    
    .rx-box-mites-culture {
        background-color: #ffffff !important;
        border-left: 5px solid #f59e0b !important;
        color: #1f2937 !important;
        border: 1px solid #fef3c7;
    }
    .rx-box-mites-culture .rx-section-label {
        color: #b45309 !important;
    }
    
    .rx-box-mites-monitor {
        background-color: #fff7ed !important;
        border-left: 5px solid #ea580c !important;
        color: #7c2d12 !important;
    }
    .rx-box-mites-monitor .rx-section-label {
        color: #c2410c !important;
    }
    
    /* Custom Bullet Items */
    .rx-list {
        margin: 0;
        padding-left: 18px;
        color: inherit;
    }
    .rx-list li {
        margin-bottom: 6px;
        line-height: 1.45;
        font-size: 0.95rem;
    }
    
    .rx-text {
        font-size: 0.98rem;
        line-height: 1.5;
        margin: 0;
        color: inherit !important;
    }
    
    /* Probability Meter Rows */
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
        font-size: 0.92rem;
        font-weight: 600;
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
        "display_name": "Healthy Robusta Leaf",
        "badge": "🟢 Healthy",
        "badge_class": "badge-healthy",
        "theme": "healthy",
        "icon": "🌿",
        "scientific_name": "Coffea canephora (Robusta)",
        "summary": "Leaf displays normal chlorosis-free tissue, intact waxy epidermis, and vigorous physiological state.",
        "treatment": {
            "immediate": "No chemical application required. Continue regular scouting routine.",
            "cultural": [
                "Maintain balanced N-P-K soil fertilization based on seasonal soil tests.",
                "Execute structural pruning (desuckering) to maintain optimal solar interception and airflow.",
                "Ensure weed-free basins around tree canopy to prevent root nutrient competition."
            ],
            "monitoring": "Conduct bi-weekly visual inspections, especially during wet or humid periods."
        }
    },
    "rust": {
        "display_name": "Coffee Leaf Rust",
        "badge": "🔴 Coffee Leaf Rust",
        "badge_class": "badge-rust",
        "theme": "rust",
        "icon": "🍂",
        "scientific_name": "Hemileia vastatrix",
        "summary": "Severe fungal pathogen causing powdery orange-yellow spore pustules on the lower leaf surface, triggering premature leaf drop and canopy dieback.",
        "treatment": {
            "immediate": "Apply protective copper fungicide (Copper Oxychloride 50% WP at 2.5–3.0 g/L) or systemic triazole (Cyproconazole/Triadimefon) if incidence exceeds 5%.",
            "cultural": [
                "Prune heavily infected lower branches and collect fallen diseased leaves; burn or bury them away from plots.",
                "Thin shade trees to promote rapid morning dew drying and lower canopy micro-humidity below 80%.",
                "Apply potassium-rich organic mulch to reinforce leaf cuticle resilience against spore penetration."
            ],
            "monitoring": "Scout the lower canopy weekly during rainy months; apply preventive sprays before peak spore release."
        }
    },
    "mites": {
        "display_name": "Red Spider Mite Infestation",
        "badge": "🟠 Red Spider Mites",
        "badge_class": "badge-mites",
        "theme": "mites",
        "icon": "🕷️",
        "scientific_name": "Oligonychus coffeae",
        "summary": "Acarid mite infestation that sucks sap from the upper leaf epidermis, causing bronzing, pale chlorotic speckling, and impaired photosynthetic yield.",
        "treatment": {
            "immediate": "Apply cold-pressed Horticultural Neem Oil (0.5%–1.0%) or registered selective miticide (Abamectin 1.8% EC / Spiromesifen) thoroughly wetting upper leaf surfaces.",
            "cultural": [
                "Dampen dusty access paths or plant vegetative perimeter windbreaks to curb dry dust clouds that stimulate mite outbreaks.",
                "Avoid indiscriminate broad-spectrum pyrethroids that exterminate natural predatory phytoseiid mites.",
                "Provide supplemental irrigation or overhead sprinkler misting during prolonged dry, sunny spells."
            ],
            "monitoring": "Inspect upper leaf surfaces bi-weekly using a 10× hand lens; check sun-exposed edge borders first."
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
    2. Resizes to 224x224 using Bilinear interpolation.
    3. Normalizes pixel values: (img / 127.5) - 1.0 (range [-1, 1]).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image_resized = image.resize(TARGET_IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(image_resized, dtype=np.float32)
    img_preprocessed = (img_array / 127.5) - 1.0
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
        probabilities = np.exp(raw_probs) / np.sum(np.exp(raw_probs)) if np.max(raw_probs) > 1.0 else raw_probs
    else:
        # Realistic mock predictions for UI testing
        img_bytes = image.tobytes()
        seed_val = sum(img_bytes[:100]) if img_bytes else 42
        rng = np.random.default_rng(seed_val)
        simulated = rng.dirichlet(np.array([2.0, 2.0, 2.0]))
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
# UI Component: Agronomic Prescription Card Renderer
# ==============================================================================
THEME_PALETTES = {
    "healthy": {
        "card_bg": "#f0fdf4",
        "card_border": "#22c55e",
        "header_color": "#14532d",
        "divider_color": "#86efac",
        "action_bg": "#dcfce7",
        "action_border": "#16a34a",
        "action_label": "#15803d",
        "action_text": "#14532d",
        "culture_bg": "#ffffff",
        "culture_border": "#22c55e",
        "culture_label": "#15803d",
        "culture_text": "#1f2937",
        "monitor_bg": "#e0f2fe",
        "monitor_border": "#0284c7",
        "monitor_label": "#0369a1",
        "monitor_text": "#0c4a6e",
    },
    "rust": {
        "card_bg": "#fff1f2",
        "card_border": "#f43f5e",
        "header_color": "#881337",
        "divider_color": "#fecdd3",
        "action_bg": "#ffe4e6",
        "action_border": "#e11d48",
        "action_label": "#be123c",
        "action_text": "#881337",
        "culture_bg": "#ffffff",
        "culture_border": "#f43f5e",
        "culture_label": "#be123c",
        "culture_text": "#1f2937",
        "monitor_bg": "#fef2f2",
        "monitor_border": "#b91c1c",
        "monitor_label": "#991b1b",
        "monitor_text": "#7f1d1d",
    },
    "mites": {
        "card_bg": "#fffbeb",
        "card_border": "#f59e0b",
        "header_color": "#78350f",
        "divider_color": "#fde68a",
        "action_bg": "#fef3c7",
        "action_border": "#d97706",
        "action_label": "#b45309",
        "action_text": "#78350f",
        "culture_bg": "#ffffff",
        "culture_border": "#f59e0b",
        "culture_label": "#b45309",
        "culture_text": "#1f2937",
        "monitor_bg": "#fff7ed",
        "monitor_border": "#ea580c",
        "monitor_label": "#c2410c",
        "monitor_text": "#7c2d12",
    },
}

def render_prescription_card(meta: dict):
    """
    Renders high-contrast, theme-colored agronomic prescription cards.
    Fully readable in both Dark Mode and Light Mode with zero low-contrast text.
    Uses inline styles and st.html to guarantee 100% theme resilience.
    """
    import textwrap
    theme = meta.get("theme", "healthy")
    palette = THEME_PALETTES.get(theme, THEME_PALETTES["healthy"])
    treatment = meta["treatment"]
    icon = meta["icon"]
    title = meta["display_name"]

    cultural_items_html = "".join(
        f'<li style="margin-bottom: 7px; color: {palette["culture_text"]}; line-height: 1.45; font-size: 0.95rem;">{item}</li>'
        for item in treatment["cultural"]
    )

    card_html = textwrap.dedent(f"""
<div style="background-color: {palette['card_bg']}; border: 2.5px solid {palette['card_border']}; border-radius: 14px; padding: 22px; margin-top: 18px; box-shadow: 0 4px 14px rgba(0,0,0,0.06); font-family: system-ui, -apple-system, sans-serif;">
    <div style="display: flex; align-items: center; gap: 10px; font-size: 1.25rem; font-weight: 800; color: {palette['header_color']}; border-bottom: 2px dashed {palette['divider_color']}; padding-bottom: 12px; margin-bottom: 16px;">
        <span style="font-size: 1.5rem;">{icon}</span>
        <span>🌾 Agronomic Prescription & Action Plan: {title}</span>
    </div>
    
    <!-- 1. Immediate Action -->
    <div style="background-color: {palette['action_bg']}; border-left: 5px solid {palette['action_border']}; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px;">
        <div style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px; color: {palette['action_label']}; margin-bottom: 6px;">
            ⚡ Immediate Action Required:
        </div>
        <p style="font-size: 0.98rem; font-weight: 700; color: {palette['action_text']}; margin: 0; line-height: 1.5;">
            {treatment['immediate']}
        </p>
    </div>
    
    <!-- 2. Cultural & Canopy Management -->
    <div style="background-color: {palette['culture_bg']}; border-left: 5px solid {palette['culture_border']}; border: 1px solid {palette['card_border']}40; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px;">
        <div style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px; color: {palette['culture_label']}; margin-bottom: 8px;">
            🌱 Cultural & Canopy Management:
        </div>
        <ul style="margin: 0; padding-left: 20px;">
            {cultural_items_html}
        </ul>
    </div>
    
    <!-- 3. Monitoring Protocol -->
    <div style="background-color: {palette['monitor_bg']}; border-left: 5px solid {palette['monitor_border']}; border-radius: 8px; padding: 14px 16px;">
        <div style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px; color: {palette['monitor_label']}; margin-bottom: 6px;">
            🔍 Monitoring & Scouting Protocol:
        </div>
        <p style="font-size: 0.95rem; color: {palette['monitor_text']}; margin: 0; line-height: 1.5;">
            {treatment['monitoring']}
        </p>
    </div>
</div>
""").strip()

    if hasattr(st, "html"):
        st.html(card_html)
    else:
        st.markdown(card_html, unsafe_allow_html=True)


# ==============================================================================
# UI Component: Sidebar
# ==============================================================================
def render_sidebar(is_demo: bool, load_error: str):
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?auto=format&fit=crop&w=400&q=80", use_container_width=True)
        st.title("🌿 Robusta AI Diagnostics")
        st.markdown(
            "Precision agronomy assistant for **Robusta (*Coffea canephora*)** farmers, agronomists, and field scouts."
        )

        st.divider()
        st.subheader("📡 Inference Engine Status")
        if is_demo:
            st.warning("⚠️ **DEMO PREVIEW MODE**")
            st.caption(
                "Trained weights (`coffee_disease_model.keras`) not found locally. "
                "Simulated predictions are active for UI testing."
            )
            if load_error:
                with st.expander("Details"):
                    st.code(load_error, language="text")
        else:
            st.success("🟢 **LIVE MODEL LOADED**")
            st.caption("MobileNetV2 neural network weights active in memory.")

        st.divider()
        st.subheader("📁 Test Samples Guide")
        st.caption(
            "You can test immediately using ready samples in the `test_samples/` folder:\n"
            "• `sample_healthy_1.jpg`\n"
            "• `sample_healthy_2.jpg`\n"
            "• `sample_rust_1.jpg`\n"
            "• `sample_rust_2.jpg`"
        )

        st.divider()
        st.subheader("📖 Target Pathologies")
        with st.expander("🟢 Healthy Robusta"):
            st.caption("Normal emerald foliage, intact cuticle, no fungal sporulation or chlorotic stippling.")
        with st.expander("🔴 Coffee Leaf Rust (*Hemileia vastatrix*)"):
            st.caption("Bright orange-yellow powdery pustules on leaf underside causing severe defoliation.")
        with st.expander("🟠 Red Spider Mite (*Oligonychus coffeae*)"):
            st.caption("Upper leaf bronzing and yellow speckled lesions from sap-sucking spider mites.")

        st.divider()
        st.caption("Robusta AI • v2.0 • High-Contrast Edition")


# ==============================================================================
# Main Application Flow
# ==============================================================================
def main():
    model, is_demo, load_error = load_disease_model()
    class_to_idx, idx_to_class = load_class_indices()

    render_sidebar(is_demo, load_error)

    # Hero Header
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">☕ Robusta Coffee Leaf Disease Diagnosis AI</div>
            <div class="hero-subtitle">
                Automated multi-class pathology classification and expert agronomic treatment prescriptions 
                for Robusta (<em>Coffea canephora</em>) plantations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_demo:
        st.info(
            "💡 **Running in Demo Mode**: To enable live deep learning inference, download `coffee_disease_model.keras` "
            "from Kaggle into this project folder. Test samples are ready in the `test_samples/` directory.",
            icon="ℹ️"
        )

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Leaf Photos (Single or Multiple)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Supports JPG, JPEG, and PNG images. For optimal results, ensure the leaf is clearly focused."
    )

    if not uploaded_files:
        st.markdown("---")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("### 📸 1. Capture")
            st.write("Take a clear, well-lit photo of the upper or lower surface of a Robusta coffee leaf.")
        with col_c2:
            st.markdown("### 🧠 2. Analyze")
            st.write("MobileNetV2 neural network detects fungal sporulation, chlorosis, and mite damage.")
        with col_c3:
            st.markdown("### 🌾 3. Prescribe")
            st.write("Receive immediate IPM chemical recommendations and long-term canopy management protocols.")

        # Quick Test Helper Section
        st.markdown("---")
        st.markdown("#### 🧪 Want a quick test? Use sample files:")
        sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
        with sample_col1:
            st.code("test_samples/sample_healthy_1.jpg", language="text")
        with sample_col2:
            st.code("test_samples/sample_healthy_2.jpg", language="text")
        with sample_col3:
            st.code("test_samples/sample_rust_1.jpg", language="text")
        with sample_col4:
            st.code("test_samples/sample_rust_2.jpg", language="text")
        return

    st.markdown(f"### 📋 Diagnostic Reports ({len(uploaded_files)} leaf image{'s' if len(uploaded_files) > 1 else ''})")

    # Process each uploaded image
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        col_img, col_diag = st.columns([1, 1.35], gap="large")

        # 1. Image Preview & Validation
        try:
            image = Image.open(uploaded_file)
            image.verify()
            image = Image.open(uploaded_file)
        except Exception as e:
            with col_img:
                st.error(f"❌ Corrupt or unreadable image: {uploaded_file.name}")
                st.caption(f"Error: {str(e)}")
            continue

        with col_img:
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            st.caption(f"Dimensions: {image.size[0]} × {image.size[1]} px | Mode: {image.mode}")

        # 2. Run Inference
        with col_diag:
            with st.spinner("Analyzing pathology..."):
                pred_class, confidence, probs = predict_disease(image, model, is_demo, idx_to_class)

            meta = CLASS_METADATA.get(pred_class, CLASS_METADATA["healthy"])
            conf_percent = confidence * 100.0

            # Classification Header with Badge & Metrics
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
                    <span class="badge-chip {meta['badge_class']}">{meta['badge']}</span>
                    <span style="font-size: 1.35rem; font-weight: 800;">
                        Confidence: <span style="color: {'#16a34a' if conf_percent >= 80 else '#d97706' if conf_percent >= 60 else '#dc2626'};">{conf_percent:.1f}%</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"**Pathogen / Agent:** *{meta['scientific_name']}*")
            st.markdown(f"<p style='font-size: 0.95rem; line-height: 1.5;'>{meta['summary']}</p>", unsafe_allow_html=True)

            # Low Confidence Warning Alert
            if confidence < CONFIDENCE_THRESHOLD_WARNING:
                st.warning(
                    f"⚠️ **Low Confidence Alert ({conf_percent:.1f}%)**: The prediction is below the 60% confidence threshold. "
                    "Ensure the leaf is in focus and well-lit, or inspect for atypical co-infections.",
                    icon="⚠️"
                )

            # Probability Breakdown
            st.markdown("##### 📊 Pathology Probability Distribution:")
            class_colors = {
                "healthy": "#10b981",
                "rust": "#f43f5e",
                "mites": "#f59e0b"
            }
            for c_name in ["healthy", "rust", "mites"]:
                c_prob = probs.get(c_name, 0.0)
                pct = c_prob * 100.0
                st.markdown(
                    f"""
                    <div class="prob-row">
                        <span>{CLASS_METADATA[c_name]['badge']}</span>
                        <span><strong>{pct:.1f}%</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(min(max(c_prob, 0.0), 1.0)))

            # Agronomic Prescription & Action Plan
            render_prescription_card(meta)


if __name__ == "__main__":
    main()
