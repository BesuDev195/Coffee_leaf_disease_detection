"""
process_custom_data.py
Preprocesses, standardizes, resizes (224x224), labels, and augments raw coffee leaf photos
from 'raw_ds/' into the structured 'dataset_custom/' format.
Supports degree-based rotation augmentation (at least 4 degree types: 0°, 90°, 180°, 270°).
"""

import os
import sys
import glob
import json
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from PIL import Image, ImageOps


def compute_file_hash(filepath: Path) -> str:
    """Computes MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_safely(filepath: Path) -> Image.Image:
    """
    Loads an image from filepath with fallback support for AVIF via ffmpeg.
    Standardizes color space to RGB and corrects EXIF orientation.
    """
    img = None
    # 1. Try standard Pillow loading
    try:
        img = Image.open(filepath)
        img.load()
    except Exception as e:
        # 2. Check for AVIF or formats unsupported by default Pillow
        if filepath.suffix.lower() == '.avif':
            try:
                cmd = [
                    'ffmpeg', '-y', '-i', str(filepath),
                    '-vframes', '1', '-f', 'image2pipe',
                    '-vcodec', 'png', '-'
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                stdout_data, _ = proc.communicate()
                if proc.returncode == 0 and stdout_data:
                    import io
                    img = Image.open(io.BytesIO(stdout_data))
                    img.load()
            except Exception as ffmpeg_err:
                raise RuntimeError(f"Failed to decode AVIF with ffmpeg: {ffmpeg_err}") from e

        if img is None:
            raise e

    # 3. Correct EXIF orientation (rotates portrait smartphone photos upright)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # 4. Standardize color channels to RGB
    if img.mode != 'RGB':
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            alpha_img = img.convert('RGBA')
            bg = Image.new('RGB', alpha_img.size, (255, 255, 255))
            bg.paste(alpha_img, mask=alpha_img.split()[3])
            img = bg
        else:
            img = img.convert('RGB')

    return img


def rotate_image_by_degree(img: Image.Image, degree: int) -> Image.Image:
    """
    Rotates an image by specified degree (0, 90, 180, 270, or arbitrary angle).
    Uses lossless transpose for orthogonal angles (90, 180, 270) to avoid artifacts.
    """
    normalized_degree = degree % 360
    if normalized_degree == 0:
        return img.copy()
    elif normalized_degree == 90:
        return img.transpose(Image.Transpose.ROTATE_270)  # Clockwise 90
    elif normalized_degree == 180:
        return img.transpose(Image.Transpose.ROTATE_180)
    elif normalized_degree == 270:
        return img.transpose(Image.Transpose.ROTATE_90)   # Clockwise 270 / CCW 90
    else:
        # Arbitrary rotation
        return img.rotate(normalized_degree, resample=Image.Resampling.BICUBIC, expand=True)


def resize_image(img: Image.Image, target_size=(224, 224), mode='direct') -> Image.Image:
    """
    Resizes image to target_size (224, 224).
    Modes:
      - 'direct': Direct high-quality Lanczos resize (standard for MobileNetV2/CNNs)
      - 'letterbox': Preserves aspect ratio, pads with border to 224x224
    """
    if mode == 'direct':
        return img.resize(target_size, Image.Resampling.LANCZOS)
    elif mode == 'letterbox':
        target_w, target_h = target_size
        img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        offset_x = (target_w - img.width) // 2
        offset_y = (target_h - img.height) // 2
        new_img.paste(img, (offset_x, offset_y))
        return new_img
    else:
        raise ValueError(f"Unknown resize mode: {mode}")


def process_dataset(
    raw_dir='raw_ds',
    output_dir='dataset_custom',
    target_size=(224, 224),
    resize_mode='direct',
    rotation_degrees=(0, 90, 180, 270),
    zip_output=True
):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    target_classes = ['healthy', 'mites', 'rust']
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff'}
    degrees = list(rotation_degrees)

    print("=" * 75)
    print("🌿 ROBUSTA COFFEE LEAF CUSTOM DATA PREPROCESSOR & DEGREE AUGMENTER")
    print("=" * 75)
    print(f"📁 Input Directory:     {raw_path.resolve()}")
    print(f"📁 Output Directory:    {out_path.resolve()}")
    print(f"📐 Target Image Size:   {target_size[0]}x{target_size[1]}")
    print(f"🎨 Resize Mode:         {resize_mode}")
    print(f"🏷️ Target Classes:      {target_classes}")
    print(f"🔄 Rotation Degrees:    {degrees} ({len(degrees)} orientation variants per photo)")
    print("=" * 75)

    if not raw_path.exists():
        print(f"❌ Error: Input directory '{raw_dir}' does not exist!")
        sys.exit(1)

    # Clean / prepare output directory
    if out_path.exists():
        print(f"🧹 Clearing existing output folder: '{output_dir}'")
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for cls_name in target_classes:
        (out_path / cls_name).mkdir(parents=True, exist_ok=True)

    # 1. Discover all raw files and check for duplicates across classes
    file_hashes = {}      # hash -> (class, filename)
    cross_duplicates = [] # duplicates found across different classes
    intra_duplicates = [] # duplicates within same class

    records = []
    category_stats = {
        cls_name: {'raw': 0, 'augmented_total': 0, 'skipped': 0}
        for cls_name in target_classes
    }

    for cls_name in target_classes:
        class_folder = raw_path / cls_name
        if not class_folder.exists():
            print(f"⚠️ Warning: Subfolder '{raw_dir}/{cls_name}' was not found. Skipping class.")
            continue

        raw_files = [f for f in class_folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        category_stats[cls_name]['raw'] = len(raw_files)

        print(f"\n🔄 Processing & Augmenting '{cls_name}' ({len(raw_files)} raw files found)...")

        idx = 1
        for f in sorted(raw_files):
            f_hash = compute_file_hash(f)

            if f_hash in file_hashes:
                prev_cls, prev_name = file_hashes[f_hash]
                if prev_cls != cls_name:
                    cross_duplicates.append((cls_name, f.name, prev_cls, prev_name))
                else:
                    intra_duplicates.append((cls_name, f.name, prev_name))
            else:
                file_hashes[f_hash] = (cls_name, f.name)

            # Load original image safely
            try:
                orig_img = load_image_safely(f)
                orig_w, orig_h = orig_img.size
                orig_fmt = f.suffix.lower()
            except Exception as e:
                print(f"   ⚠️ Could not load '{f.name}': {e}")
                category_stats[cls_name]['skipped'] += 1
                continue

            # Generate all degree rotations
            for deg in degrees:
                try:
                    rotated_img = rotate_image_by_degree(orig_img, deg)
                    processed_img = resize_image(rotated_img, target_size=target_size, mode=resize_mode)

                    # Standardized filename: custom_<class>_<0001>_deg<deg>.jpg
                    new_filename = f"custom_{cls_name}_{idx:04d}_deg{deg:03d}.jpg"
                    dest_file = out_path / cls_name / new_filename

                    # Save as clean JPEG (quality=95)
                    processed_img.save(dest_file, format='JPEG', quality=95)

                    # Record metadata & RoCoLe-compatible one-hot labels
                    records.append({
                        'filename': new_filename,
                        'filepath': str(dest_file.relative_to(out_path.parent)),
                        'class': cls_name,
                        'class_idx': target_classes.index(cls_name),
                        'rotation_deg': deg,
                        'healthy': 1 if cls_name == 'healthy' else 0,
                        'mites': 1 if cls_name == 'mites' else 0,
                        'rust': 1 if cls_name == 'rust' else 0,
                        'orig_filename': f.name,
                        'orig_width': orig_w,
                        'orig_height': orig_h,
                        'orig_format': orig_fmt,
                    })

                    category_stats[cls_name]['augmented_total'] += 1

                except Exception as e:
                    print(f"   ⚠️ Could not process rotation {deg}° for '{f.name}': {e}")

            idx += 1

    # 2. Output Annotations CSV & Dataset Summary JSON
    csv_path = out_path / 'annotations.csv'
    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"\n📄 Saved RoCoLe-compatible annotations CSV: {csv_path.resolve()}")

    summary = {
        'target_size': target_size,
        'resize_mode': resize_mode,
        'rotation_degrees': degrees,
        'num_degree_variants': len(degrees),
        'classes': target_classes,
        'total_processed': len(records),
        'category_stats': category_stats,
        'cross_class_duplicates_count': len(cross_duplicates),
        'intra_class_duplicates_count': len(intra_duplicates)
    }

    summary_path = out_path / 'dataset_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # 3. Print Results Summary
    print("\n" + "=" * 75)
    print("📊 DATASET PROCESSING & AUGMENTATION SUMMARY")
    print("=" * 75)
    print(f"{'Class':<12} {'Raw Found':<12} {'Augmented Total (224x224)':<28} {'Skipped':<10}")
    print("-" * 75)
    for cls_name in target_classes:
        stats = category_stats[cls_name]
        print(f"{cls_name:<12} {stats['raw']:<12} {stats['augmented_total']:<28} {stats['skipped']:<10}")
    print("-" * 75)
    print(f"Total Output Images: {len(records)} ({len(degrees)}x multiplier across {len(records)//max(1, len(degrees))} raw photos)")

    # 4. Alert on Duplicates
    if cross_duplicates:
        print("\n" + "!" * 75)
        print("⚠️ CRITICAL WARNING: CROSS-CLASS DUPLICATES DETECTED!")
        print("!" * 75)
        print(f"Found {len(cross_duplicates)} exact duplicate image(s) shared across different classes:")
        for cur_c, cur_f, prev_c, prev_f in cross_duplicates[:5]:
            print(f"   • '{cur_c}/{cur_f}' is identical to '{prev_c}/{prev_f}'")
        if len(cross_duplicates) > 5:
            print(f"   ... and {len(cross_duplicates) - 5} more identical files.")
        print("💡 NOTE: If you copied 'rust/' to 'mites/' as a placeholder, please ensure")
        print("   you replace 'raw_ds/mites/' with real spider mite photos before training!")
        print("!" * 75)

    # 5. Optional Zip Archive for Kaggle Upload
    if zip_output and len(records) > 0:
        zip_base_name = str(out_path)
        print(f"\n📦 Packaging '{output_dir}' into '{zip_base_name}.zip' for easy Kaggle upload...")
        shutil.make_archive(zip_base_name, 'zip', root_dir=out_path)
        print(f"✅ Archive created: {Path(zip_base_name + '.zip').resolve()}")

    print("\n🎉 Done! Your augmented & standardized dataset is ready in:")
    print(f"   👉 {out_path.resolve()}/")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Process, augment by degrees, and label custom coffee leaf photos.")
    parser.add_argument('--input_dir', default='raw_ds', help="Path to raw images folder (default: raw_ds)")
    parser.add_argument('--output_dir', default='dataset_custom', help="Path to processed output folder (default: dataset_custom)")
    parser.add_argument('--size', type=int, default=224, help="Target image width and height (default: 224)")
    parser.add_argument('--mode', choices=['direct', 'letterbox'], default='direct',
                        help="Resize mode: 'direct' (stretch/standard CNN) or 'letterbox' (pad with borders)")
    parser.add_argument('--degrees', nargs='+', type=int, default=[0, 90, 180, 270],
                        help="List of rotation degrees to generate (default: 0 90 180 270)")
    parser.add_argument('--no-zip', dest='zip_output', action='store_false', help="Do not create a zip archive")

    args = parser.parse_args()
    process_dataset(
        raw_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=(args.size, args.size),
        resize_mode=args.mode,
        rotation_degrees=args.degrees,
        zip_output=args.zip_output
    )


if __name__ == '__main__':
    main()
