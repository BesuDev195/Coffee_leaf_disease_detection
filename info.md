# ☕ Coffee Leaf Disease Detection — System Information & Class Imbalance Strategy

---

## 🔗 Quick Links
- **Kaggle Training Notebook**: [https://www.kaggle.com/code/hyq3rfly/notebook17d0048b1f/edit](https://www.kaggle.com/code/hyq3rfly/notebook17d0048b1f/edit)

---

## 📊 Combined Dataset Breakdown

The training pipeline merges **Source A (RoCoLe)** and **Source B (Custom Field Photos)** into a clean, unified dataset:

| Pathology Class | Source A (RoCoLe) | Source B (Custom Augmented) | Total Combined Images | Share (%) |
|---|:---:|:---:|:---:|:---:|
| **Healthy** | 791 | 232 (augmented from 58 raw) | **1,023** | 49.0% |
| **Rust** (*Hemileia vastatrix*) | 602 | 296 (augmented from 74 raw) | **898** | 43.0% |
| **Mites** (*Oligonychus coffeae*) | 167 | 0 (pure authentic RoCoLe) | **167** | 8.0% |
| **Total** | **1,560** | **528** | **2,088** | **100%** |

> **Key Architectural Decision**: 
> Removing the placeholder duplicate images from `raw_ds/mites/` leaves 167 authentic, expert-verified Robusta spider mite leaf photos from RoCoLe. This eliminates all conflicting duplicate signals between Rust and Mites.

---

## ⚖️ How the System Solves Class Imbalance

With 1,023 Healthy and 898 Rust vs. 167 Mites (~6:1 ratio), naive training would risk favoring the majority classes. Our pipeline solves this through four integrated mechanisms:

### 1. Mathematical Inverse Class Weighting (Notebook Step 9)
The loss function applies balanced class weights calculated via Scikit-Learn:

$$W_j = \frac{N_{\text{total}}}{C \times N_j}$$

Where:
- $N_{\text{total}} = 2,088$ (total training samples)
- $C = 3$ (number of classes)
- $N_j$ = count of samples in class $j$

#### Computed Weighting Table:
| Class | Sample Count ($N_j$) | Loss Weight ($W_j$) | Training Effect |
|---|:---:|:---:|---|
| **Healthy** | 1,023 | **0.68** | Standard backpropagation penalty |
| **Rust** | 898 | **0.78** | Standard backpropagation penalty |
| **Mites** | 167 | **4.17** | **~6.1× Higher Penalty for Errors!** |

**Plain-English Meaning**:  
During model optimization, any prediction error made on a **Mites** leaf penalizes the neural network **~6 times more heavily** than an error on a Healthy leaf. The optimizer is mathematically compelled to learn the subtle bronze stippling and damage patterns of mites despite the smaller sample count.

---

### 2. On-the-Fly Dynamic Data Augmentation (Notebook Step 7)
Even though there are 167 base mite images, the network **never sees the identical image twice**:
- On every epoch, the `tf.data` pipeline randomly applies:
  - `RandomFlip("horizontal_and_vertical")`
  - `RandomRotation(0.20)` (up to $\pm 72^\circ$)
  - `RandomZoom(0.15)`
  - `RandomContrast(0.10)`
- Across 20 training epochs, the model processes **over 2,000 unique synthetic variations** of spider mite damage angles, scales, and lighting conditions.

---

### 3. Stratified Dataset Splitting (Notebook Step 6)
To avoid test set starvation, the raw file paths are partitioned using `stratify=labels`:
- **Training Set (70%)**: $\approx 1,461$ images ($\approx 117$ mites)
- **Validation Set (15%)**: $\approx 313$ images ($\approx 25$ mites)
- **Test Set (15%)**: $\approx 314$ images ($\approx 25$ mites)

This guarantees consistent class distributions across all three splits before batches are created.

---

### 4. Minority-Sensitive Evaluation Metrics (Notebook Step 12)
The notebook rejects raw overall accuracy as the sole success metric (which can mask minority failures) and instead evaluates:
- **Per-Class Recall**: Measures the percentage of actual mite leaves correctly caught (aiming for $\ge 85\%$).
- **Per-Class Precision**: Measures how many predicted mite leaves are true mites.
- **Confusion Matrix Heatmap**: Visualizes true vs. predicted counts across all 9 combinations to immediately spot any cross-class confusion.

---

## 🛠️ Project Execution Quick Reference

```bash
# 1. Activate Virtual Environment
source venv/bin/activate   # Linux/macOS
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

# 2. Re-preprocess custom photos if adding new raw leaves
python process_custom_data.py --input_dir raw_ds --output_dir dataset_custom --size 224

# 3. Launch Local Streamlit Web App
streamlit run app.py
```
