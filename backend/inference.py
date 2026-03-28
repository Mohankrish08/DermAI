# ==========================================
# inference.py
# Local Inference using Trained Weights
# ==========================================
#
# Usage:
#   Single image:
#     python inference.py --image path/to/image.jpg --age 45 --sex male --loc back
#
#   Batch (from CSV):
#     python inference.py --batch --csv metadata_balanced.csv --images HAM10000_images
#
#   Use DP model instead of FL model:
#     python inference.py --image path/to/image.jpg --age 45 --sex male --loc back --model final_DP_model.pth
# ==========================================

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import pandas as pd

import config
from backend.model import EfficientNet_ViT_Metadata


# ==========================================
# Constants
# ==========================================

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

DISEASE_FULL_NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesion",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}

SEX_MAP = {"male": 0, "female": 1, "unknown": 2}

# Localization mapping — must match the LabelEncoder order
# used during training. These are sorted alphabetically
# (sklearn LabelEncoder default).
LOCALIZATION_LIST = sorted([
    "abdomen", "acral", "back", "chest", "ear", "face",
    "foot", "genital", "hand", "lower extremity", "neck",
    "scalp", "trunk", "upper extremity", "unknown"
])
LOC_MAP = {loc: idx for idx, loc in enumerate(LOCALIZATION_LIST)}


# ==========================================
# Image Transform (same as val_transform)
# ==========================================

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])


# ==========================================
# Load Model
# ==========================================

def load_model(weights_path, device):
    model = EfficientNet_ViT_Metadata(num_classes=config.NUM_CLASSES)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ Loaded model weights from: {weights_path}")
    return model


# ==========================================
# Encode Metadata
# ==========================================

def encode_metadata(age, sex, localization):
    """
    Encode a single patient's metadata into a tensor [3].
    Matches the encoding used during training.
    """
    age_norm = age / 100.0

    sex_lower = sex.strip().lower() if sex else "unknown"
    sex_enc = SEX_MAP.get(sex_lower, 2)

    loc_lower = localization.strip().lower() if localization else "unknown"
    loc_enc = LOC_MAP.get(loc_lower, LOC_MAP.get("unknown", 0))

    return torch.tensor([age_norm, sex_enc, loc_enc], dtype=torch.float32)


# ==========================================
# Single Image Prediction
# ==========================================

def predict_single(model, image_path, age, sex, localization, device, top_k=3):
    """
    Predict on a single dermoscopic image + metadata.
    Returns top-k class predictions with probabilities.
    """
    transform = get_inference_transform()

    # Load & transform image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)       # [1, 3, 224, 224]

    # Encode metadata
    meta_tensor = encode_metadata(age, sex, localization).unsqueeze(0).to(device)  # [1, 3]

    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor, meta_tensor)                # [1, 7]
        probs = F.softmax(logits, dim=1).squeeze(0)            # [7]

    # Top-k
    topk_probs, topk_indices = torch.topk(probs, k=min(top_k, len(CLASS_NAMES)))

    results = []
    for prob, idx in zip(topk_probs, topk_indices):
        cls_name = CLASS_NAMES[idx.item()]
        results.append({
            "class": cls_name,
            "disease": DISEASE_FULL_NAMES[cls_name],
            "probability": prob.item()
        })

    return results


# ==========================================
# Batch Prediction (from CSV)
# ==========================================

def predict_batch(model, csv_path, img_dir, device):
    """
    Run predictions on all rows in a CSV file.
    CSV must have: image_id, age, sex, localization
    """
    transform = get_inference_transform()
    df = pd.read_csv(csv_path)

    all_results = []

    for _, row in df.iterrows():
        image_id = row["image_id"]

        # Find image
        img_path = os.path.join(img_dir, image_id + ".jpg")
        if not os.path.exists(img_path):
            print(f"⚠️  Skipping {image_id}: image not found")
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        age = row.get("age", 50)
        sex = str(row.get("sex", "unknown"))
        loc = str(row.get("localization", "unknown"))

        meta_tensor = encode_metadata(age, sex, loc).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor, meta_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        pred_idx = probs.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]
        pred_prob = probs[pred_idx].item()

        true_label = row.get("dx", "N/A")

        all_results.append({
            "image_id": image_id,
            "true_label": true_label,
            "predicted": pred_class,
            "confidence": pred_prob,
        })

    return pd.DataFrame(all_results)


# ==========================================
# Pretty Print
# ==========================================

def print_prediction(results, image_path):
    print("\n" + "=" * 55)
    print(f"🔬 SKIN LESION CLASSIFICATION RESULT")
    print(f"   Image: {os.path.basename(image_path)}")
    print("=" * 55)

    for i, r in enumerate(results):
        bar_len = int(r["probability"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ◀ PREDICTED" if i == 0 else ""
        print(f"  {r['class']:>6s}  |{bar}| {r['probability']:.1%}  {r['disease']}{marker}")

    print("=" * 55)
    print()


# ==========================================
# CLI Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification — Inference",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Mode
    parser.add_argument("--batch", action="store_true",
                        help="Run batch prediction from a CSV file")

    # Single-image args
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single dermoscopic image (.jpg)")
    parser.add_argument("--age", type=float, default=50.0,
                        help="Patient age (default: 50)")
    parser.add_argument("--sex", type=str, default="unknown",
                        choices=["male", "female", "unknown"],
                        help="Patient sex (default: unknown)")
    parser.add_argument("--loc", type=str, default="unknown",
                        help="Body localization (e.g., back, face, trunk)")

    # Batch args
    parser.add_argument("--csv", type=str, default="metadata_balanced.csv",
                        help="CSV file for batch prediction")
    parser.add_argument("--images", type=str, default="HAM10000_images",
                        help="Image directory for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV for batch results")

    # Model
    parser.add_argument("--model", type=str, default="final_FL_model.pth",
                        help="Path to model weights (.pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu or cuda (default: auto-detect)")

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🖥  Device: {device}")

    # Load model
    if not os.path.exists(args.model):
        print(f"❌ Model weights not found: {args.model}")
        print("   Download from Colab or train first.")
        sys.exit(1)

    model = load_model(args.model, device)

    # ── Batch mode ──
    if args.batch:
        print(f"\n📂 Batch prediction from: {args.csv}")
        results_df = predict_batch(model, args.csv, args.images, device)
        results_df.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to: {args.output}")
        print(results_df.head(20).to_string(index=False))
        return

    # ── Single-image mode ──
    if not args.image:
        print("❌ Provide --image <path> for single prediction or --batch for CSV.")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    results = predict_single(
        model, args.image,
        age=args.age, sex=args.sex, localization=args.loc,
        device=device, top_k=3
    )

    print_prediction(results, args.image)


if __name__ == "__main__":
    main()
