# ☕ Robusta Coffee Leaf Disease Detection & Diagnosis AI System

An end-to-end Computer Vision system for diagnosing foliar diseases and pests in **Robusta Coffee (*Coffea canephora*)** leaves using Transfer Learning with **MobileNetV2**.

---

## 🎯 Target Pathology Classes

| Class ID | Target Label | Pathogen / Agent | Description |
|:---:|:---|:---|:---|
| **0** | `healthy` | None | Clean, symptom-free Robusta leaf tissue |
| **1** | `mites` | *Oligonychus coffeae* | Red Spider Mite infestation causing bronze stippling |
| **2** | `rust` | *Hemileia vastatrix* | Coffee Leaf Rust (all severity levels 1–4 merged) |

---

## 📁 Project Structure

```
coffee/
├── app.py                         # Interactive local Streamlit web application
├── class_indices.json             # Class-to-index mapping (updated by notebook)
├── generate_notebook.py           # Standalone Python script to generate the notebook
├── README.md                      # Comprehensive documentation and setup guide
├── requirements.txt               # Pinned local dependencies
└── rocole_coffee_disease_cnn.ipynb # Complete training pipeline (generated)
```

---

## 🚀 Step 1: Environment Setup

Activate your existing Python virtual environment and install the required dependencies:

### On Windows (PowerShell / Command Prompt):
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# Command Prompt (cmd)
.\venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### On macOS / Linux (bash / zsh):
```bash
source venv/bin/activate
pip install -r requirements.txt
```

> **Note**: Cloud platforms like Kaggle already have TensorFlow, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-Learn preinstalled.

---

## 📓 Step 2: Generate the Training Notebook

Run the standalone generator script to create the Jupyter notebook:

```bash
python generate_notebook.py
```

This creates `rocole_coffee_disease_cnn.ipynb` containing all alternating markdown explanations and executable code cells.

---

## 🗂️ Step 3: Dataset Ingestion & Organization

The system fuses two data sources inside the notebook:

### Source A: RoCoLe Dataset (Robusta Coffee Leaf)
- **Source**: Available on Kaggle (Search: `"RoCoLe"` or `"Robusta Coffee Leaf Disease"`).
- **Structure**: Contains leaf photographs along with an annotations CSV file indicating disease status (`healthy`, `rust` severity levels, and `red spider mite`).
- **Auto-Scanner**: The notebook automatically searches `/kaggle/input` (or local subfolders) for the CSV and images, maps all rust levels to `"rust"`, red spider mite to `"mites"`, and healthy to `"healthy"`.

### Source B: Custom Field Photos (`dataset_custom/`)
If you have captured your own Robusta leaf photos, organize them as follows:

```
dataset_custom/
├── healthy/
│   ├── leaf_001.jpg
│   └── leaf_002.png
├── rust/
│   ├── rust_001.jpg
│   └── rust_002.jpeg
└── mites/
    ├── mite_001.jpg
    └── mite_002.png
```

- Supported extensions: `.jpg`, `.jpeg`, `.png` (case-insensitive).
- Corrupted images are automatically detected with PIL and skipped.
- If you do not have custom photos yet, Source B is optional; the pipeline runs seamlessly with RoCoLe alone.

---

## 🏋️ Step 4: Model Training Options

### Option A: Cloud GPU Training on Kaggle (Recommended — Free GPU)

1. Open [Kaggle](https://www.kaggle.com/) and click **"+ New Notebook"**.
2. **Enable GPU Accelerator**:
   - In the right-hand panel under **Notebook Options** ➔ **Accelerator**, select **GPU T4 x2** or **GPU P100**.
3. **Attach RoCoLe Dataset**:
   - Click **"+ Add Data"** in the top right.
   - Search for **"RoCoLe"** and click **Add**.
4. **Attach Custom Photos (Optional)**:
   - Click **"+ Add Data"** ➔ **"Upload a Dataset"** to upload your zipped `dataset_custom/` as a private dataset.
5. **Upload the Notebook**:
   - In the Kaggle notebook menu, go to **File** ➔ **Upload Notebook** and select `rocole_coffee_disease_cnn.ipynb`.
6. **Run Training**:
   - Click **"Run All"**.
   - Training runs Stage 1 (frozen backbone) followed by Stage 2 (fine-tuning top layers).
7. **Download Outputs**:
   - Once execution finishes, inspect the **Output** panel (`/kaggle/working/`).
   - Download the following files:
     - `coffee_disease_model.keras`
     - `class_indices.json`
     - `training_curves.png`
     - `confusion_matrix.png`

---

### Option B: Local Training in VS Code / Jupyter

1. Place the RoCoLe dataset and/or `dataset_custom/` in the same directory as the notebook.
2. Launch VS Code or Jupyter Notebook:
   ```bash
   jupyter notebook rocole_coffee_disease_cnn.ipynb
   ```
3. Select your virtual environment kernel.
4. Execute all cells from top to bottom.
5. The model checkpoint `coffee_disease_model.keras` and `class_indices.json` will be saved directly to your project folder.

---

## 💻 Step 5: Launch the Streamlit Web App

1. Ensure the downloaded model weights and index files are placed in the same folder as `app.py`:
   ```
   coffee/
   ├── app.py
   ├── class_indices.json
   └── coffee_disease_model.keras
   ```
2. Start the local Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.

### App Features:
- **Instant Diagnosis**: Drag and drop single or multiple leaf photos.
- **Classification & Probabilities**: Colored badges (🟢 Healthy, 🔴 Rust, 🟠 Mites) with full probability breakdowns.
- **Agronomic Prescriptions**: Immediate remedial action, canopy management, and scouting protocols.
- **Low-Confidence Alert**: Warns when confidence is below 60% to prevent misdiagnosis in ambiguous field conditions.
- **Demo Mode**: If `coffee_disease_model.keras` is not yet downloaded, the app launches in Demo Mode with simulated predictions.

---

## 🛠️ Troubleshooting Guide

### 1. "Demo Mode Active" Banner in Streamlit
- **Cause**: `app.py` cannot locate `coffee_disease_model.keras` in the working directory.
- **Fix**: Verify that `coffee_disease_model.keras` is in the exact same folder from which you ran `streamlit run app.py`.

### 2. "No GPU detected. Running on CPU" in Notebook
- **Cause**: On Kaggle, the notebook accelerator is set to "None".
- **Fix**: Open the right sidebar in Kaggle ➔ **Settings** ➔ **Accelerator** ➔ Choose **GPU T4 x2** or **P100**, and restart the session.

### 3. RoCoLe Auto-Scan Error: "RoCoLe dataset was not detected"
- **Cause**: The RoCoLe dataset was not attached or placed under a different directory name.
- **Fix**: On Kaggle, click **"+ Add Data"**, search for RoCoLe, and add it. Locally, ensure the folder containing the images and CSV is inside the project workspace.

### 4. Class Order Mismatch
- **Cause**: Hardcoded index ordering differing between training and inference.
- **Fix**: `app.py` reads `class_indices.json` generated directly during the notebook's export step. Always copy both `coffee_disease_model.keras` and `class_indices.json` together.
