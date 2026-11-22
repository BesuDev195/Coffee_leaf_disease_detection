import os
import glob
from pathlib import Path
import tensorflow as tf

print("=" * 65)
print("🚀 KAGGLE SYSTEM & DATASET HEALTH CHECK")
print("=" * 65)

# 1. Check GPU Accelerator
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU Online: {len(gpus)} GPU(s) available")
    for gpu in gpus:
        print(f"   • {gpu.name}")
else:
    print("⚠️ No GPU detected! Go to Settings -> Accelerator -> select GPU T4 x2.")

# 2. Check Write Permission to /kaggle/working
try:
    test_file = "/kaggle/working/_test.tmp"
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)
    print("✅ Storage: /kaggle/working is writable")
except Exception as e:
    print(f"❌ Storage error: {e}")

# 3. Check Mounted Datasets in /kaggle/input
print("\n📁 Checking Datasets in /kaggle/input:")

# Scan Custom Dataset
custom_healthy = len(glob.glob("/kaggle/input/**/custom/**/healthy/*.jpg", recursive=True) + glob.glob("/kaggle/input/**/healthy/custom_*.jpg", recursive=True))
custom_rust = len(glob.glob("/kaggle/input/**/custom/**/rust/*.jpg", recursive=True) + glob.glob("/kaggle/input/**/rust/custom_*.jpg", recursive=True))
custom_total = custom_healthy + custom_rust

# Scan RoCoLe Dataset
rocole_healthy = len(glob.glob("/kaggle/input/**/coffee___healthy/*.jpg", recursive=True))
rocole_rust = len(glob.glob("/kaggle/input/**/coffee___rust/*.jpg", recursive=True))
rocole_mites = len(glob.glob("/kaggle/input/**/coffee___red_spider_mite/*.jpg", recursive=True))
rocole_total = rocole_healthy + rocole_rust + rocole_mites

print(f"   • RoCoLe Dataset:      {rocole_total:>4} images ({rocole_healthy} healthy, {rocole_rust} rust, {rocole_mites} mites)")
print(f"   • Custom Dataset:      {custom_total:>4} images ({custom_healthy} healthy, {custom_rust} rust, 0 duplicate mites)")
print(f"   -----------------------------------------------")
print(f"   • Total Available:     {rocole_total + custom_total:>4} clean images")

# 4. Final Verdict
print("\n" + "=" * 65)
if gpus and rocole_total > 0 and custom_total > 0:
    print("🎉 ALL SYSTEMS GO! Your Kaggle session is 100% READY.")
    print("👉 You can now click 'Run All' to start training!")
else:
    print("⚠️ Check the items above marked with ⚠️ before starting training.")
print("=" * 65)