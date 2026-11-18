"""
generate_notebook.py
Generates the complete, self-contained Jupyter notebook:
'rocole_coffee_disease_cnn.ipynb'
using nbformat.
"""

import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # =========================================================================
    # 1. Title and Introduction
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "# ☕ Robusta Coffee Leaf Disease Detection & Diagnosis AI System\n\n"
        "Welcome! This notebook builds an end-to-end Computer Vision pipeline to detect and diagnose "
        "diseases in **Robusta Coffee (*Coffea canephora*)** leaves.\n\n"
        "### 🎯 Target Classes:\n"
        "1. **healthy** — Clean, disease-free coffee leaves\n"
        "2. **rust** — Coffee Leaf Rust (*Hemileia vastatrix*), including all severity levels\n"
        "3. **mites** — Red Spider Mite (*Oligonychus coffeae*) infestation\n\n"
        "### 🔬 System Architecture:\n"
        "- **Dual-Source Ingestion**: Merges the public **RoCoLe dataset** and your **private manual photos**.\n"
        "- **Data Pipeline**: Strict stratified split (70/15/15) before batching; on-the-fly data augmentation.\n"
        "- **Model**: Transfer Learning with **MobileNetV2** (ImageNet pre-trained) + Custom Classification Head.\n"
        "- **Two-Stage Training**: Frozen feature extractor followed by fine-tuning of top convolutional layers.\n"
        "- **Evaluation**: Test set metrics, confusion matrix heatmap, and training curves.\n"
        "- **Export**: Produces `coffee_disease_model.keras` and `class_indices.json` for deployment in the Streamlit web app."
    ))

    # =========================================================================
    # 2. Imports and Environment Detection
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 🛠️ Step 1: Imports & Hardware Acceleration Detection\n\n"
        "We detect whether the notebook is running inside **Kaggle** (`/kaggle/input` exists) or on a **local machine**.\n"
        "If a compatible GPU is detected, we automatically enable **mixed precision** (`mixed_float16`) for faster training and reduced memory consumption."
    ))

    cells.append(nbf.v4.new_code_cell(
        'import os\n'
        'import sys\n'
        'import glob\n'
        'import json\n'
        'import shutil\n'
        'import random\n'
        'import warnings\n'
        'from pathlib import Path\n'
        '\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import matplotlib.pyplot as plt\n'
        'import seaborn as sns\n'
        'from PIL import Image\n'
        'from sklearn.model_selection import train_test_split\n'
        'from sklearn.utils import class_weight\n'
        'from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n'
        '\n'
        'import tensorflow as tf\n'
        'from tensorflow.keras import layers, models, optimizers, callbacks\n'
        'from tensorflow.keras.applications import MobileNetV2\n'
        'from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n'
        '\n'
        'warnings.filterwarnings("ignore")\n'
        '\n'
        '# 1. Detect execution environment\n'
        'IS_KAGGLE = os.path.exists("/kaggle/input")\n'
        'ENV_NAME = "Kaggle Cloud" if IS_KAGGLE else "Local Machine / VS Code"\n'
        'print(f"\\n🌐 Execution Environment: {ENV_NAME}")\n'
        'print(f"🐍 Python Version: {sys.version.split()[0]}")\n'
        'print(f"📦 TensorFlow Version: {tf.__version__}")\n'
        '\n'
        '# 2. Detect Hardware Acceleration (GPU / CPU)\n'
        'gpus = tf.config.list_physical_devices("GPU")\n'
        'if gpus:\n'
        '    print(f"🚀 Detected {len(gpus)} GPU(s):")\n'
        '    for gpu in gpus:\n'
        '        print(f"   • {gpu.name}")\n'
        '    # Enable mixed precision for modern GPUs (computes in float16, keeps weights in float32)\n'
        '    try:\n'
        '        tf.keras.mixed_precision.set_global_policy("mixed_float16")\n'
        '        print("⚡ Mixed precision policy set to: mixed_float16")\n'
        '    except Exception as e:\n'
        '        print(f"⚠️ Could not set mixed precision: {e}")\n'
        'else:\n'
        '    print("💻 No GPU detected. Running on CPU (training may take longer).")\n'
    ))

    # =========================================================================
    # 3. Global Configuration
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## ⚙️ Step 2: Global Configuration & Hyperparameters\n\n"
        "All file paths, random seeds, training hyperparameters, and fine-tuning options are defined in this single configuration cell."
    ))

    cells.append(nbf.v4.new_code_cell(
        '# =============================================================================\n'
        '# GLOBAL HYPERPARAMETERS & PATHS CONFIGURATION\n'
        '# =============================================================================\n'
        'class Config:\n'
        '    # Random reproducibility seed\n'
        '    SEED = 42\n'
        '    \n'
        '    # Image dimensions expected by MobileNetV2\n'
        '    IMG_HEIGHT = 224\n'
        '    IMG_WIDTH = 224\n'
        '    CHANNELS = 3\n'
        '    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)\n'
        '    \n'
        '    # Batching and Epochs\n'
        '    BATCH_SIZE = 32\n'
        '    STAGE_1_EPOCHS = 15     # Feature extractor training (frozen base)\n'
        '    STAGE_2_EPOCHS = 10     # Fine-tuning top convolutional layers\n'
        '    FINE_TUNE = True        # Set False to skip Stage 2 fine-tuning\n'
        '    FINE_TUNE_LAYERS = 30   # Unfreeze last N layers during Stage 2\n'
        '    \n'
        '    # Learning Rates\n'
        '    STAGE_1_LR = 1e-3\n'
        '    STAGE_2_LR = 1e-5\n'
        '    \n'
        '    # Data Split Proportions (Sum = 1.0)\n'
        '    TRAIN_SPLIT = 0.70\n'
        '    VAL_SPLIT = 0.15\n'
        '    TEST_SPLIT = 0.15\n'
        '    \n'
        '    # Target Classes (Alphabetical order ensures stable indexing)\n'
        '    CLASSES = ["healthy", "mites", "rust"]\n'
        '    NUM_CLASSES = len(CLASSES)\n'
        '    \n'
        '    # Directories\n'
        '    BASE_INPUT_DIR = "/kaggle/input" if IS_KAGGLE else "."\n'
        '    OUTPUT_DIR = "/kaggle/working" if IS_KAGGLE else "."\n'
        '    MERGED_DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset_unified")\n'
        '    \n'
        '    # Output Artifact File Names\n'
        '    MODEL_FILENAME = "coffee_disease_model.keras"\n'
        '    CLASS_INDICES_FILENAME = "class_indices.json"\n'
        '    CURVES_PLOT_FILENAME = "training_curves.png"\n'
        '    CONF_MATRIX_FILENAME = "confusion_matrix.png"\n'
        '\n'
        '# Seed everything for deterministic reproducibility\n'
        'random.seed(Config.SEED)\n'
        'np.random.seed(Config.SEED)\n'
        'tf.random.set_seed(Config.SEED)\n'
        '\n'
        'print(f"✅ Configuration initialized:")\n'
        'print(f"   • Image Size: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}x{Config.CHANNELS}")\n'
        'print(f"   • Batch Size: {Config.BATCH_SIZE}")\n'
        'print(f"   • Output Directory: {os.path.abspath(Config.OUTPUT_DIR)}")\n'
        'print(f"   • Classes ({Config.NUM_CLASSES}): {Config.CLASSES}")\n'
    ))

    # =========================================================================
    # 4. Auto-scan Source A (RoCoLe Dataset)
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 🔍 Step 3: Automatic Scanner for Source A (RoCoLe Dataset)\n\n"
        "The **RoCoLe (Robusta Coffee Leaf)** dataset contains photos and an annotations CSV file.\n"
        "Instead of guessing file paths or column names, our helper function recursively scans the input directory, "
        "locates all images and CSV files, detects the annotation columns, and standardizes labels to:\n"
        "- `healthy` ➔ Healthy\n"
        "- `rust` (any severity level 1–4) ➔ Rust\n"
        "- `red spider mite` ➔ Mites\n\n"
        "If RoCoLe is not yet attached, a friendly guide explains exactly what to do."
    ))

    cells.append(nbf.v4.new_code_cell(
        'def scan_rocole_dataset(search_dir):\n'
        '    """\n'
        '    Recursively inspects search_dir for RoCoLe images and annotation CSV.\n'
        '    Returns a DataFrame with columns: [\'image_path\', \'label\', \'source\'].\n'
        '    """\n'
        '    print(f"🔍 Scanning \'{search_dir}\' for RoCoLe files...")\n'
        '    valid_exts = {".jpg", ".jpeg", ".png"}\n'
        '    \n'
        '    # 1. Discover all image files\n'
        '    image_paths = []\n'
        '    for root, _, files in os.walk(search_dir):\n'
        '        # Skip merged dataset directory to prevent self-scanning\n'
        '        if "dataset_unified" in root:\n'
        '            continue\n'
        '        for f in files:\n'
        '            if Path(f).suffix.lower() in valid_exts:\n'
        '                image_paths.append(os.path.join(root, f))\n'
        '                \n'
        '    print(f"   Found {len(image_paths)} total images in \'{search_dir}\'.")\n'
        '    \n'
        '    # Quick index: filename -> full path\n'
        '    name_to_path = {os.path.basename(p): p for p in image_paths}\n'
        '    name_no_ext_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}\n'
        '    \n'
        '    # 2. Discover all CSV files\n'
        '    csv_paths = []\n'
        '    for root, _, files in os.walk(search_dir):\n'
        '        if "dataset_unified" in root:\n'
        '            continue\n'
        '        for f in files:\n'
        '            if f.lower().endswith(".csv"):\n'
        '                csv_paths.append(os.path.join(root, f))\n'
        '                \n'
        '    print(f"   Found {len(csv_paths)} CSV file(s): {[os.path.basename(c) for c in csv_paths]}")\n'
        '    \n'
        '    records = []\n'
        '    matched_csv = None\n'
        '    \n'
        '    # 3. Analyze CSV files to find RoCoLe annotations\n'
        '    for cpath in csv_paths:\n'
        '        try:\n'
        '            df = pd.read_csv(cpath)\n'
        '        except Exception as e:\n'
        '            print(f"   ⚠️ Could not read {cpath}: {e}")\n'
        '            continue\n'
        '            \n'
        '        cols_lower = [c.lower() for c in df.columns]\n'
        '        \n'
        '        # Look for filename column\n'
        '        fn_col = None\n'
        '        for col in df.columns:\n'
        '            sample_vals = df[col].astype(str).head(10).tolist()\n'
        '            if any(val in name_to_path or val in name_no_ext_to_path for val in sample_vals):\n'
        '                fn_col = col\n'
        '                break\n'
        '            if any(c_kw in col.lower() for c_kw in ["filename", "image", "photo", "name", "file"]):\n'
        '                fn_col = col\n'
        '                break\n'
        '                \n'
        '        if not fn_col:\n'
        '            continue\n'
        '            \n'
        '        # Try parsing RoCoLe labels from rows\n'
        '        print(f"   Testing candidate annotation CSV: \'{os.path.basename(cpath)}\' (Image col: \'{fn_col}\')")\n'
        '        \n'
        '        # Check if RoCoLe uses individual disease columns or a single diagnosis column\n'
        '        # Common RoCoLe schema has columns: [\'Healthy\', \'Rust\', \'Red Spider Mite\'] or categorical column\n'
        '        has_rust_col = any("rust" in c for c in cols_lower)\n'
        '        has_mite_col = any("mite" in c or "spider" in c for c in cols_lower)\n'
        '        has_healthy_col = any("healthy" in c for c in cols_lower)\n'
        '        \n'
        '        found_any = False\n'
        '        for _, row in df.iterrows():\n'
        '            fname = str(row[fn_col]).strip()\n'
        '            # Resolve image file path\n'
        '            full_img_path = name_to_path.get(fname)\n'
        '            if not full_img_path:\n'
        '                full_img_path = name_no_ext_to_path.get(os.path.splitext(fname)[0])\n'
        '            if not full_img_path:\n'
        '                continue\n'
        '                \n'
        '            label = None\n'
        '            # Strategy A: Multi-column indicators (e.g. Healthy, Rust, Spider Mite / Red Spider Mite)\n'
        '            if has_rust_col or has_mite_col or has_healthy_col:\n'
        '                for col in df.columns:\n'
        '                    val = row[col]\n'
        '                    c_low = col.lower()\n'
        '                    # Numeric non-zero or True\n'
        '                    is_pos = (isinstance(val, (int, float)) and val > 0) or str(val).lower() in ["1", "true", "yes", "positive"]\n'
        '                    if is_pos:\n'
        '                        if "rust" in c_low:\n'
        '                            label = "rust"\n'
        '                            break\n'
        '                        elif "mite" in c_low or "spider" in c_low:\n'
        '                            label = "mites"\n'
        '                            break\n'
        '                        elif "healthy" in c_low:\n'
        '                            label = "healthy"\n'
        '            \n'
        '            # Strategy B: Single diagnosis/label column\n'
        '            if not label:\n'
        '                for col in df.columns:\n'
        '                    val_str = str(row[col]).lower()\n'
        '                    if "rust" in val_str:\n'
        '                        label = "rust"\n'
        '                        break\n'
        '                    elif "mite" in val_str or "spider" in val_str:\n'
        '                        label = "mites"\n'
        '                        break\n'
        '                    elif "healthy" in val_str:\n'
        '                        label = "healthy"\n'
        '                        break\n'
        '                        \n'
        '            if label in Config.CLASSES:\n'
        '                records.append({\n'
        '                    "image_path": full_img_path,\n'
        '                    "label": label,\n'
        '                    "source": "rocole"\n'
        '                })\n'
        '                found_any = True\n'
        '                \n'
        '        if found_any:\n'
        '            matched_csv = cpath\n'
        '            break\n'
        '            \n'
        '    # Fallback Strategy C: If images were organized by subfolder already\n'
        '    if not records:\n'
        '        print("   No CSV mapping succeeded. Checking for folder-based class structure...")\n'
        '        for p in image_paths:\n'
        '            parent = os.path.basename(os.path.dirname(p)).lower()\n'
        '            if "rust" in parent:\n'
        '                records.append({"image_path": p, "label": "rust", "source": "rocole"})\n'
        '            elif "mite" in parent or "spider" in parent:\n'
        '                records.append({"image_path": p, "label": "mites", "source": "rocole"})\n'
        '            elif "healthy" in parent:\n'
        '                records.append({"image_path": p, "label": "healthy", "source": "rocole"})\n'
        '                \n'
        '    df_rocole = pd.DataFrame(records)\n'
        '    if len(df_rocole) > 0:\n'
        '        print(f"✅ RoCoLe Auto-Scan Successful! Found {len(df_rocole)} labeled images.")\n'
        '        if matched_csv:\n'
        '            print(f"   Annotation CSV: {os.path.abspath(matched_csv)}")\n'
        '        print("   Counts per class:")\n'
        '        print(df_rocole["label"].value_counts().to_string())\n'
        '    else:\n'
        '        err_msg = (\n'
        '            "\\n❌ RoCoLe dataset was not detected in search path!\\n\\n"\n'
        '            "HOW TO FIX ON KAGGLE:\\n"\n'
        '            "1. Click \'+ Add Data\' (top right panel).\\n"\n'
        '            "2. Search for \'RoCoLe\' or \'Robusta Coffee Leaf\' and click Add.\\n\\n"\n'
        '            "HOW TO FIX LOCALLY:\\n"\n'
        '            "1. Download RoCoLe from Kaggle (or extract the zip into this folder).\\n"\n'
        '            "2. Ensure the images and CSV annotations file are inside a subfolder here.\\n"\n'
        '        )\n'
        '        print(err_msg)\n'
        '        \n'
        '    return df_rocole\n'
        '\n'
        '# Execute RoCoLe auto-scan\n'
        'df_rocole = scan_rocole_dataset(Config.BASE_INPUT_DIR)\n'
    ))

    # =========================================================================
    # 5. Merge Source A + Source B
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 📦 Step 4: Scan Source B (Custom Dataset) & Merge into Unified Dataset\n\n"
        "We now scan for **Source B** (`dataset_custom/healthy/`, `dataset_custom/rust/`, `dataset_custom/mites/`).\n"
        "Both sources are copied into a clean unified directory (`dataset_unified/`) with unique prefixes (`rocole_`, `custom_`).\n"
        "During this process, we:\n"
        "- Verify each file with PIL to safely drop corrupt or unreadable images.\n"
        "- Display a cross-tabulation table of counts by source and class.\n"
        "- Detect and display a warning if there is heavy class imbalance."
    ))

    cells.append(nbf.v4.new_code_cell(
        'def scan_custom_dataset(search_dir):\n'
        '    """\n'
        '    Scans search_dir for custom photos structured under healthy/, rust/, mites/.\n'
        '    """\n'
        '    print(f"🔍 Scanning \'{search_dir}\' for Source B (Custom Dataset)...")\n'
        '    valid_exts = {".jpg", ".jpeg", ".png"}\n'
        '    records = []\n'
        '    \n'
        '    for root, _, files in os.walk(search_dir):\n'
        '        if "dataset_unified" in root:\n'
        '            continue\n'
        '        parent = os.path.basename(root).lower()\n'
        '        label = None\n'
        '        if "healthy" in parent:\n'
        '            label = "healthy"\n'
        '        elif "rust" in parent:\n'
        '            label = "rust"\n'
        '        elif "mite" in parent or "spider" in parent:\n'
        '            label = "mites"\n'
        '            \n'
        '        if label in Config.CLASSES:\n'
        '            for f in files:\n'
        '                if Path(f).suffix.lower() in valid_exts:\n'
        '                    records.append({\n'
        '                        "image_path": os.path.join(root, f),\n'
        '                        "label": label,\n'
        '                        "source": "custom"\n'
        '                    })\n'
        '                    \n'
        '    df_custom = pd.DataFrame(records)\n'
        '    if len(df_custom) > 0:\n'
        '        print(f"✅ Found {len(df_custom)} custom images:")\n'
        '        print(df_custom["label"].value_counts().to_string())\n'
        '    else:\n'
        '        print("ℹ️ Note: No custom photos found in \'dataset_custom/\' (Source B is optional).")\n'
        '    return df_custom\n'
        '\n'
        'df_custom = scan_custom_dataset(Config.BASE_INPUT_DIR)\n'
        '\n'
        '# Combine sources\n'
        'dfs_to_concat = [df for df in [df_rocole, df_custom] if len(df) > 0]\n'
        'if not dfs_to_concat:\n'
        '    raise FileNotFoundError(\n'
        '        "❌ Neither Source A (RoCoLe) nor Source B (Custom) contains valid images!\\n"\n'
        '        "Please ensure at least one dataset is provided."\n'
        '    )\n'
        '\n'
        'df_all = pd.concat(dfs_to_concat, ignore_index=True)\n'
        '\n'
        '# Build unified directory structure\n'
        'if os.path.exists(Config.MERGED_DATASET_DIR):\n'
        '    shutil.rmtree(Config.MERGED_DATASET_DIR)\n'
        'for cls_name in Config.CLASSES:\n'
        '    os.makedirs(os.path.join(Config.MERGED_DATASET_DIR, cls_name), exist_ok=True)\n'
        '\n'
        'unified_records = []\n'
        'corrupt_count = 0\n'
        '\n'
        'print("\\n🚚 Unifying dataset and checking for corrupt images...")\n'
        'for idx, row in df_all.iterrows():\n'
        '    src_path = row["image_path"]\n'
        '    cls_name = row["label"]\n'
        '    src_tag = row["source"]\n'
        '    \n'
        '    # Integrity check with PIL\n'
        '    try:\n'
        '        with Image.open(src_path) as img:\n'
        '            img.verify()  # Fast check for corruption\n'
        '        with Image.open(src_path) as img:\n'
        '            img.convert("RGB")  # Ensure readable as 3-channel RGB\n'
        '    except Exception as e:\n'
        '        corrupt_count += 1\n'
        '        continue\n'
        '        \n'
        '    # Copy with unique prefix\n'
        '    base_name = os.path.basename(src_path)\n'
        '    dest_name = f"{src_tag}_{idx:05d}_{base_name}"\n'
        '    dest_path = os.path.join(Config.MERGED_DATASET_DIR, cls_name, dest_name)\n'
        '    shutil.copy2(src_path, dest_path)\n'
        '    \n'
        '    unified_records.append({\n'
        '        "file_path": dest_path,\n'
        '        "label": cls_name,\n'
        '        "source": src_tag\n'
        '    })\n'
        '\n'
        'df_unified = pd.DataFrame(unified_records)\n'
        'print(f"\\n📊 Unification Complete! Valid images: {len(df_unified)} (Corrupt skipped: {corrupt_count})")\n'
        '\n'
        '# Summary Cross-Tabulation Table\n'
        'cross_tab = pd.crosstab(df_unified["label"], df_unified["source"], margins=True)\n'
        'print("\\n📋 Dataset Breakdown (Class × Source):")\n'
        'print(cross_tab)\n'
        '\n'
        '# Check for heavy class imbalance (ratio > 2.0)\n'
        'counts = df_unified["label"].value_counts()\n'
        'imbalance_ratio = counts.max() / counts.min()\n'
        'if imbalance_ratio > 2.0:\n'
        '    print(f"\\n⚠️ WARNING: Heavy class imbalance detected! (Ratio: {imbalance_ratio:.2f}:1)")\n'
        '    print("   Balanced class weights will be computed and applied during training.")\n'
        'else:\n'
        '    print(f"\\n⚖️ Class distribution is relatively balanced (Ratio: {imbalance_ratio:.2f}:1).")\n'
    ))

    # =========================================================================
    # 6. Sample Images Visualization
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 🖼️ Step 5: Visualizing Sample Images\n\n"
        "Let's inspect sample leaf photos from each of our three target classes."
    ))

    cells.append(nbf.v4.new_code_cell(
        'plt.figure(figsize=(12, 4))\n'
        'for i, cls_name in enumerate(Config.CLASSES):\n'
        '    cls_files = df_unified[df_unified["label"] == cls_name]["file_path"].tolist()\n'
        '    if not cls_files:\n'
        '        continue\n'
        '    sample_path = random.choice(cls_files)\n'
        '    img = Image.open(sample_path)\n'
        '    \n'
        '    plt.subplot(1, Config.NUM_CLASSES, i + 1)\n'
        '    plt.imshow(img)\n'
        '    plt.title(f"Class: {cls_name.capitalize()}\\n({len(cls_files)} samples)", fontsize=12, fontweight="bold")\n'
        '    plt.axis("off")\n'
        '    \n'
        'plt.tight_layout()\n'
        'plt.show()\n'
    ))

    # =========================================================================
    # 7. Stratified Dataset Split (No Leakage)
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## ✂️ Step 6: Stratified Train / Validation / Test Split\n\n"
        "To strictly avoid **data leakage**, we perform stratified splitting on the raw file paths **before** "
        "constructing `tf.data` input pipelines:\n"
        "- **Train (70%)**: Used to fit model weights\n"
        "- **Validation (15%)**: Used for hyperparameter tuning, checkpointing, and early stopping\n"
        "- **Test (15%)**: Held out strictly for final unbiased evaluation"
    ))

    cells.append(nbf.v4.new_code_cell(
        '# Create numeric label mapping\n'
        'class_to_idx = {name: idx for idx, name in enumerate(Config.CLASSES)}\n'
        'idx_to_class = {idx: name for idx, name in enumerate(Config.CLASSES)}\n'
        '\n'
        'filepaths = df_unified["file_path"].values\n'
        'labels_str = df_unified["label"].values\n'
        'labels_num = np.array([class_to_idx[l] for l in labels_str])\n'
        '\n'
        '# 1. Split Train (70%) vs Temp (30%)\n'
        'train_files, temp_files, train_y, temp_y = train_test_split(\n'
        '    filepaths,\n'
        '    labels_num,\n'
        '    test_size=(Config.VAL_SPLIT + Config.TEST_SPLIT),\n'
        '    stratify=labels_num,\n'
        '    random_state=Config.SEED\n'
        ')\n'
        '\n'
        '# 2. Split Temp (30%) equally into Val (15%) and Test (15%)\n'
        'val_files, test_files, val_y, test_y = train_test_split(\n'
        '    temp_files,\n'
        '    temp_y,\n'
        '    test_size=0.50,\n'
        '    stratify=temp_y,\n'
        '    random_state=Config.SEED\n'
        ')\n'
        '\n'
        'print(f"📊 Dataset Split Summary:")\n'
        'print(f"   • Training:   {len(train_files):>4} samples ({len(train_files)/len(filepaths):.1%})")\n'
        'print(f"   • Validation: {len(val_files):>4} samples ({len(val_files)/len(filepaths):.1%})")\n'
        'print(f"   • Test:       {len(test_files):>4} samples ({len(test_files)/len(filepaths):.1%})")\n'
        'print(f"   • Total:      {len(filepaths):>4} samples")\n'
    ))

    # =========================================================================
    # 8. tf.data Pipelines
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## ⚡ Step 7: High-Performance tf.data Pipelines with Augmentation\n\n"
        "We construct scalable `tf.data.Dataset` pipelines that:\n"
        "1. Read and decode images from disk in parallel.\n"
        "2. Resize images to `224x224` and one-hot encode labels.\n"
        "3. Apply **data augmentation** (random flip, rotation, zoom, contrast) to the **training set only**.\n"
        "4. Prefetch batches using `tf.data.AUTOTUNE` to eliminate I/O GPU stalls."
    ))

    cells.append(nbf.v4.new_code_cell(
        'def load_and_preprocess_image(path_tensor, label_tensor):\n'
        '    """Loads an image from path, decodes as RGB, and resizes to 224x224."""\n'
        '    img_bytes = tf.io.read_file(path_tensor)\n'
        '    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)\n'
        '    img = tf.image.resize(img, [Config.IMG_HEIGHT, Config.IMG_WIDTH])\n'
        '    img = tf.cast(img, tf.float32)\n'
        '    # One-hot encode label for categorical crossentropy\n'
        '    label_one_hot = tf.one_hot(label_tensor, depth=Config.NUM_CLASSES)\n'
        '    return img, label_one_hot\n'
        '\n'
        '# Keras Data Augmentation layer (applied to training pipeline only)\n'
        'data_augmentation = tf.keras.Sequential([\n'
        '    layers.RandomFlip("horizontal_and_vertical"),\n'
        '    layers.RandomRotation(0.2),\n'
        '    layers.RandomZoom(0.15),\n'
        '    layers.RandomContrast(0.1),\n'
        '], name="data_augmentation")\n'
        '\n'
        'AUTOTUNE = tf.data.AUTOTUNE\n'
        '\n'
        'def build_pipeline(files, labels, is_training=False):\n'
        '    ds = tf.data.Dataset.from_tensor_slices((files, labels))\n'
        '    if is_training:\n'
        '        ds = ds.shuffle(buffer_size=len(files), seed=Config.SEED)\n'
        '    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)\n'
        '    ds = ds.batch(Config.BATCH_SIZE)\n'
        '    if is_training:\n'
        '        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)\n'
        '    ds = ds.prefetch(buffer_size=AUTOTUNE)\n'
        '    return ds\n'
        '\n'
        'train_ds = build_pipeline(train_files, train_y, is_training=True)\n'
        'val_ds = build_pipeline(val_files, val_y, is_training=False)\n'
        'test_ds = build_pipeline(test_files, test_y, is_training=False)\n'
        '\n'
        'print("✅ tf.data pipelines built successfully.")\n'
    ))

    # =========================================================================
    # 9. Model Architecture
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 🧠 Step 8: Transfer Learning Model Architecture (MobileNetV2)\n\n"
        "We leverage **MobileNetV2** pre-trained on ImageNet:\n"
        "- Lightweight and ideal for edge devices / local Streamlit deployment.\n"
        "- Includes native `preprocess_input` (scales pixel range $[0, 255] \\rightarrow [-1, 1]$).\n"
        "- Custom Head: `GlobalAveragePooling2D` ➔ `Dropout(0.3)` ➔ `Dense(128, ReLU)` ➔ `Dense(3, Softmax)`.\n"
        "- We force `float32` output activation for numerical stability under mixed precision."
    ))

    cells.append(nbf.v4.new_code_cell(
        'def create_coffee_disease_model():\n'
        '    # 1. Input Layer\n'
        '    inputs = layers.Input(shape=Config.INPUT_SHAPE, name="leaf_image_input")\n'
        '    \n'
        '    # 2. Preprocessing Layer (MobileNetV2 expects [-1, 1])\n'
        '    x = preprocess_input(inputs)\n'
        '    \n'
        '    # 3. Backbone: MobileNetV2 (Feature Extractor frozen in Stage 1)\n'
        '    base_model = MobileNetV2(\n'
        '        input_shape=Config.INPUT_SHAPE,\n'
        '        include_top=False,\n'
        '        weights="imagenet"\n'
        '    )\n'
        '    base_model.trainable = False\n'
        '    x = base_model(x, training=False)\n'
        '    \n'
        '    # 4. Custom Classification Head\n'
        '    x = layers.GlobalAveragePooling2D(name="gap")(x)\n'
        '    x = layers.BatchNormalization(name="head_bn")(x)\n'
        '    x = layers.Dropout(0.3, name="head_dropout")(x)\n'
        '    x = layers.Dense(128, activation="relu", name="dense_128")(x)\n'
        '    x = layers.Dropout(0.2, name="head_dropout_2")(x)\n'
        '    \n'
        '    # 5. Output Layer (forced to float32 for mixed precision stability)\n'
        '    outputs = layers.Dense(\n'
        '        Config.NUM_CLASSES,\n'
        '        activation="softmax",\n'
        '        dtype="float32",\n'
        '        name="predictions"\n'
        '    )(x)\n'
        '    \n'
        '    model = models.Model(inputs=inputs, outputs=outputs, name="CoffeeLeafDiseaseDetector")\n'
        '    return model, base_model\n'
        '\n'
        'model, base_model = create_coffee_disease_model()\n'
        'model.summary()\n'
    ))

    # =========================================================================
    # 10. Class Weights Computation
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## ⚖️ Step 9: Computing Class Weights\n\n"
        "To ensure the model does not become biased toward the most frequent class, we calculate "
        "balanced class weights using scikit-learn's `compute_class_weight`."
    ))

    cells.append(nbf.v4.new_code_cell(
        'unique_classes = np.unique(train_y)\n'
        'weights_array = class_weight.compute_class_weight(\n'
        '    class_weight="balanced",\n'
        '    classes=unique_classes,\n'
        '    y=train_y\n'
        ')\n'
        'class_weights_dict = {cls_idx: float(w) for cls_idx, w in zip(unique_classes, weights_array)}\n'
        '\n'
        'print("⚖️ Computed Class Weights for Loss Weighting:")\n'
        'for idx, w in class_weights_dict.items():\n'
        '    print(f"   • {idx_to_class[idx]:<10} (class {idx}): {w:.4f}")\n'
    ))

    # =========================================================================
    # 11. Callbacks Configuration
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 🛡️ Step 10: Callbacks (Checkpointing, Early Stopping, LR Scheduling)\n\n"
        "We set up three safety callbacks:\n"
        "- **ModelCheckpoint**: Saves the best model checkpoint (`coffee_disease_model.keras`) based on minimum `val_loss`.\n"
        "- **EarlyStopping**: Restores the best weights if `val_loss` fails to improve for 5 consecutive epochs.\n"
        "- **ReduceLROnPlateau**: Dynamically drops the learning rate when training plateaus."
    ))

    cells.append(nbf.v4.new_code_cell(
        'best_model_path = os.path.join(Config.OUTPUT_DIR, Config.MODEL_FILENAME)\n'
        '\n'
        'callback_list = [\n'
        '    callbacks.ModelCheckpoint(\n'
        '        filepath=best_model_path,\n'
        '        monitor="val_loss",\n'
        '        mode="min",\n'
        '        save_best_only=True,\n'
        '        verbose=1\n'
        '    ),\n'
        '    callbacks.EarlyStopping(\n'
        '        monitor="val_loss",\n'
        '        mode="min",\n'
        '        patience=5,\n'
        '        restore_best_weights=True,\n'
        '        verbose=1\n'
        '    ),\n'
        '    callbacks.ReduceLROnPlateau(\n'
        '        monitor="val_loss",\n'
        '        mode="min",\n'
        '        factor=0.2,\n'
        '        patience=2,\n'
        '        min_lr=1e-7,\n'
        '        verbose=1\n'
        '    )\n'
        ']\n'
        'print(f"✅ Callbacks configured. Best model checkpoint will be saved to: {best_model_path}")\n'
    ))

    # =========================================================================
    # 12. Training: Stage 1 (Frozen Backbone) & Stage 2 (Fine-Tuning)
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 🏋️ Step 11: Two-Stage Model Training\n\n"
        "1. **Stage 1 (Feature Extraction)**: The MobileNetV2 backbone is frozen. Only the classification head learns "
        "at a standard learning rate ($10^{-3}$).\n"
        "2. **Stage 2 (Fine-Tuning)**: If `FINE_TUNE = True`, the top 30 layers of the backbone are unfrozen and trained "
        "together with the head at a very gentle learning rate ($10^{-5}$) to specialize features for coffee leaf textures."
    ))

    cells.append(nbf.v4.new_code_cell(
        '# Compile Stage 1\n'
        'model.compile(\n'
        '    optimizer=optimizers.Adam(learning_rate=Config.STAGE_1_LR),\n'
        '    loss="categorical_crossentropy",\n'
        '    metrics=["accuracy"]\n'
        ')\n'
        '\n'
        'print("🚀 Starting Stage 1 Training (Frozen Backbone)...")\n'
        'history_stage1 = model.fit(\n'
        '    train_ds,\n'
        '    validation_data=val_ds,\n'
        '    epochs=Config.STAGE_1_EPOCHS,\n'
        '    class_weight=class_weights_dict,\n'
        '    callbacks=callback_list,\n'
        '    verbose=1\n'
        ')\n'
        '\n'
        'history_stage2 = None\n'
        'if Config.FINE_TUNE:\n'
        '    print("\\n🔓 Unfreezing top backbone layers for Stage 2 Fine-Tuning...")\n'
        '    base_model.trainable = True\n'
        '    # Freeze all layers except the last N layers\n'
        '    for layer in base_model.layers[:-Config.FINE_TUNE_LAYERS]:\n'
        '        layer.trainable = False\n'
        '        \n'
        '    # Recompile with very low learning rate\n'
        '    model.compile(\n'
        '        optimizer=optimizers.Adam(learning_rate=Config.STAGE_2_LR),\n'
        '        loss="categorical_crossentropy",\n'
        '        metrics=["accuracy"]\n'
        '    )\n'
        '    \n'
        '    stage1_epochs_run = len(history_stage1.history["loss"])\n'
        '    total_epochs = stage1_epochs_run + Config.STAGE_2_EPOCHS\n'
        '    \n'
        '    print(f"🚀 Starting Stage 2 Fine-Tuning (Epochs {stage1_epochs_run + 1} to {total_epochs})...")\n'
        '    history_stage2 = model.fit(\n'
        '        train_ds,\n'
        '        validation_data=val_ds,\n'
        '        initial_epoch=stage1_epochs_run,\n'
        '        epochs=total_epochs,\n'
        '        class_weight=class_weights_dict,\n'
        '        callbacks=callback_list,\n'
        '        verbose=1\n'
        '    )\n'
    ))

    # =========================================================================
    # 13. Test Evaluation & Artifact Generation
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 🧪 Step 12: Unbiased Test Set Evaluation & Curves\n\n"
        "We evaluate the saved best model checkpoint on the **unseen Test set**.\n"
        "We print the Scikit-Learn **Classification Report** and save:\n"
        "1. `training_curves.png`: Training vs. Validation Loss and Accuracy.\n"
        "2. `confusion_matrix.png`: Normalized Seaborn confusion matrix heatmap."
    ))

    cells.append(nbf.v4.new_code_cell(
        '# Load the verified best model saved by ModelCheckpoint\n'
        'print(f"📂 Loading best checkpoint from {best_model_path}...")\n'
        'eval_model = models.load_model(best_model_path)\n'
        '\n'
        '# Gather test predictions\n'
        'y_pred_probs = []\n'
        'y_true = []\n'
        '\n'
        'for batch_x, batch_y in test_ds:\n'
        '    probs = eval_model.predict(batch_x, verbose=0)\n'
        '    y_pred_probs.append(probs)\n'
        '    y_true.append(np.argmax(batch_y.numpy(), axis=1))\n'
        '\n'
        'y_pred_probs = np.vstack(y_pred_probs)\n'
        'y_pred = np.argmax(y_pred_probs, axis=1)\n'
        'y_true = np.concatenate(y_true)\n'
        '\n'
        'test_acc = accuracy_score(y_true, y_pred)\n'
        'print(f"\\n🎯 Unseen Test Set Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\\n")\n'
        '\n'
        'print("📄 Classification Report:")\n'
        'print(classification_report(y_true, y_pred, target_names=Config.CLASSES, digits=4))\n'
        '\n'
        '# 1. Plot & Save Confusion Matrix\n'
        'cm = confusion_matrix(y_true, y_pred)\n'
        'plt.figure(figsize=(7, 6))\n'
        'sns.heatmap(\n'
        '    cm,\n'
        '    annot=True,\n'
        '    fmt="d",\n'
        '    cmap="Blues",\n'
        '    xticklabels=Config.CLASSES,\n'
        '    yticklabels=Config.CLASSES,\n'
        '    cbar=False,\n'
        '    annot_kws={"size": 14, "weight": "bold"}\n'
        ')\n'
        'plt.title(f"Confusion Matrix (Test Acc: {test_acc*100:.1f}%)", fontsize=14, pad=15)\n'
        'plt.xlabel("Predicted Class", fontsize=12)\n'
        'plt.ylabel("True Class", fontsize=12)\n'
        'plt.tight_layout()\n'
        'cm_save_path = os.path.join(Config.OUTPUT_DIR, Config.CONF_MATRIX_FILENAME)\n'
        'plt.savefig(cm_save_path, dpi=300)\n'
        'plt.show()\n'
        'print(f"💾 Confusion matrix saved to: {cm_save_path}")\n'
        '\n'
        '# 2. Combine and Plot Training History Curves\n'
        'train_loss = history_stage1.history["loss"]\n'
        'val_loss = history_stage1.history["val_loss"]\n'
        'train_acc = history_stage1.history["accuracy"]\n'
        'val_acc = history_stage1.history["val_accuracy"]\n'
        '\n'
        'if history_stage2 is not None:\n'
        '    train_loss += history_stage2.history["loss"]\n'
        '    val_loss += history_stage2.history["val_loss"]\n'
        '    train_acc += history_stage2.history["accuracy"]\n'
        '    val_acc += history_stage2.history["val_accuracy"]\n'
        '\n'
        'epochs_range = range(1, len(train_loss) + 1)\n'
        '\n'
        'plt.figure(figsize=(14, 5))\n'
        '# Loss Subplot\n'
        'plt.subplot(1, 2, 1)\n'
        'plt.plot(epochs_range, train_loss, label="Training Loss", color="#1f77b4", lw=2)\n'
        'plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ff7f0e", lw=2)\n'
        'if history_stage2 is not None:\n'
        '    plt.axvline(x=len(history_stage1.history["loss"]), color="gray", linestyle="--", label="Fine-Tuning Start")\n'
        'plt.title("Categorical Cross-Entropy Loss", fontsize=13)\n'
        'plt.xlabel("Epoch", fontsize=11)\n'
        'plt.ylabel("Loss", fontsize=11)\n'
        'plt.legend()\n'
        'plt.grid(True, alpha=0.3)\n'
        '\n'
        '# Accuracy Subplot\n'
        'plt.subplot(1, 2, 2)\n'
        'plt.plot(epochs_range, train_acc, label="Training Accuracy", color="#1f77b4", lw=2)\n'
        'plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#ff7f0e", lw=2)\n'
        'if history_stage2 is not None:\n'
        '    plt.axvline(x=len(history_stage1.history["loss"]), color="gray", linestyle="--", label="Fine-Tuning Start")\n'
        'plt.title("Model Accuracy", fontsize=13)\n'
        'plt.xlabel("Epoch", fontsize=11)\n'
        'plt.ylabel("Accuracy", fontsize=11)\n'
        'plt.legend()\n'
        'plt.grid(True, alpha=0.3)\n'
        '\n'
        'plt.tight_layout()\n'
        'curves_save_path = os.path.join(Config.OUTPUT_DIR, Config.CURVES_PLOT_FILENAME)\n'
        'plt.savefig(curves_save_path, dpi=300)\n'
        'plt.show()\n'
        'print(f"💾 Training curves saved to: {curves_save_path}")\n'
    ))

    # =========================================================================
    # 14. Save class_indices.json
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 💾 Step 13: Export Class Indices Mapping\n\n"
        "We save `class_indices.json` to guarantee the Streamlit web application decodes predictions "
        "in the exact same order as training."
    ))

    cells.append(nbf.v4.new_code_cell(
        'class_indices_path = os.path.join(Config.OUTPUT_DIR, Config.CLASS_INDICES_FILENAME)\n'
        'with open(class_indices_path, "w") as f:\n'
        '    json.dump(class_to_idx, f, indent=2)\n'
        '\n'
        'print(f"✅ Class indices exported to: {class_indices_path}")\n'
        'with open(class_indices_path, "r") as f:\n'
        '    print(f.read())\n'
    ))

    # =========================================================================
    # 15. Final Summary and Deployment Instructions
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 🚀 Step 14: Deployment Checklist & Download Instructions\n\n"
        "Congratulations! Your model is trained, evaluated, and ready for deployment.\n\n"
        "### 📥 Files to Download from Kaggle Working Directory (`/kaggle/working`):\n"
        "1. **`coffee_disease_model.keras`** — Trained MobileNetV2 model weights & architecture.\n"
        "2. **`class_indices.json`** — Class mapping dictionary.\n"
        "3. **`training_curves.png`** — Loss & accuracy curves.\n"
        "4. **`confusion_matrix.png`** — Confusion matrix visualization.\n\n"
        "### 💻 Local Streamlit Setup:\n"
        "Place `coffee_disease_model.keras` and `class_indices.json` in the same directory as `app.py`:\n"
        "```bash\n"
        "coffee/\n"
        "├── app.py\n"
        "├── class_indices.json          <- Downloaded from Kaggle\n"
        "├── coffee_disease_model.keras  <- Downloaded from Kaggle\n"
        "└── ...\n"
        "```\n"
        "Then launch your interactive web application:\n"
        "```bash\n"
        "streamlit run app.py\n"
        "```"
    ))

    cells.append(nbf.v4.new_code_cell(
        'print("==================================================================")\n'
        'print("🎉 ALL TRAINING & EXPORT STEPS COMPLETED SUCCESSFULLY!")\n'
        'print("==================================================================")\n'
        'print("Artifacts generated in output directory:")\n'
        'for fname in [Config.MODEL_FILENAME, Config.CLASS_INDICES_FILENAME, Config.CURVES_PLOT_FILENAME, Config.CONF_MATRIX_FILENAME]:\n'
        '    fpath = os.path.join(Config.OUTPUT_DIR, fname)\n'
        '    if os.path.exists(fpath):\n'
        '        size_mb = os.path.getsize(fpath) / (1024 * 1024)\n'
        '        print(f"   • {fname:<28} ({size_mb:.2f} MB)")\n'
        'print("==================================================================")\n'
    ))

    nb['cells'] = cells
    return nb


def main():
    notebook_filename = "rocole_coffee_disease_cnn.ipynb"
    print(f"Building {notebook_filename} using nbformat...")
    nb = build_notebook()
    with open(notebook_filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✅ Successfully wrote {notebook_filename}")


if __name__ == "__main__":
    main()
