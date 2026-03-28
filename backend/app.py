# ==========================================
# backend/app.py
# FastAPI for Skin Lesion Classification
# ==========================================

import os
import sys
import traceback
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from backend.model import EfficientNet_ViT_Metadata

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

DISEASE_RISK = {
    "akiec": "pre-cancerous",
    "bcc":   "malignant",
    "bkl":   "benign",
    "df":    "benign",
    "mel":   "malignant",
    "nv":    "benign",
    "vasc":  "benign",
}

SEX_MAP = {"male": 0, "female": 1, "unknown": 2}

LOCALIZATION_LIST = sorted([
    "abdomen", "acral", "back", "chest", "ear", "face",
    "foot", "genital", "hand", "lower extremity", "neck",
    "scalp", "trunk", "upper extremity", "unknown"
])
LOC_MAP = {loc: idx for idx, loc in enumerate(LOCALIZATION_LIST)}

# ==========================================
# Model State
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_PATH = None


def load_model(weights_path=None):
    global MODEL, MODEL_PATH

    if weights_path is None:
        project_root = os.path.dirname(__file__)
        candidates = [
            os.path.join(project_root, "best_model.pth"),
            os.path.join(project_root, "final_FL_model.pth"),
            os.path.join(project_root, "final_DP_model.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break

    if weights_path is None or not os.path.exists(weights_path):
        print("[WARN] No model weights found. API will run but inference won't work.")
        return False

    try:
        model = EfficientNet_ViT_Metadata(num_classes=config.NUM_CLASSES)
        state = torch.load(weights_path, map_location=DEVICE, weights_only=False)

        fixed_state = {k.replace("_module.", ""): v for k, v in state.items()}

        model.load_state_dict(fixed_state, strict=False)
        print("[OK] Model loaded successfully with strict=False")
        model.to(DEVICE)
        model.eval()
        MODEL = model
        MODEL_PATH = os.path.abspath(weights_path)
        print(f"[OK] Loaded model from: {MODEL_PATH}")
        print(f"[OK] Running on device : {DEVICE}")
        return True

    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        traceback.print_exc()
        return False


# ==========================================
# Lifespan
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 55)
    print("Skin Lesion Classification API")
    print("=" * 55)
    load_model()
    yield


# ==========================================
# App Setup
# ==========================================

app = FastAPI(title="Skin Lesion Classification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Helpers
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


def encode_metadata(age, sex, localization):
    age_norm = float(age) / 100.0
    sex_enc = SEX_MAP.get(sex.strip().lower() if sex else "unknown", 2)
    loc_enc = LOC_MAP.get(localization.strip().lower() if localization else "unknown", LOC_MAP["unknown"])
    return torch.tensor([age_norm, sex_enc, loc_enc], dtype=torch.float32)


# ==========================================
# Routes
# ==========================================

static_dir = os.path.join(os.path.dirname(__file__), "static")

# Mount entire static folder
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/{full_path:path}")
def serve_angular(full_path: str):
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API is running. Frontend not deployed yet."}

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "device": str(DEVICE),
        "num_classes": config.NUM_CLASSES,
        "class_names": CLASS_NAMES,
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    age: float = Form(50.0),
    sex: str = Form("unknown"),
    localization: str = Form("unknown"),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # Validate image format
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}. Use jpg/png.")

    # Clamp age
    age = max(0.0, min(float(age), 120.0))

    # Validate sex and localization
    if sex not in SEX_MAP:
        sex = "unknown"
    if localization not in LOC_MAP:
        localization = "unknown"

    try:
        start_time = time.time()

        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        transform = get_inference_transform()
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        meta_tensor = encode_metadata(age, sex, localization).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = MODEL(img_tensor, meta_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        all_probs = {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASS_NAMES)}

        topk_probs, topk_indices = torch.topk(probs, k=len(CLASS_NAMES))
        predictions = []
        for prob, idx in zip(topk_probs, topk_indices):
            cls_name = CLASS_NAMES[idx.item()]
            predictions.append({
                "class": cls_name,
                "disease": DISEASE_FULL_NAMES[cls_name],
                "probability": round(prob.item(), 4),
                "risk": DISEASE_RISK[cls_name],
            })

        return {
            "success": True,
            "predictions": predictions,
            "all_probabilities": all_probs,
            "metadata_used": {"age": age, "sex": sex, "localization": localization},
            "inference_time_seconds": round(time.time() - start_time, 3),
            "model_path": MODEL_PATH,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/localizations")
def get_localizations():
    return {"localizations": LOCALIZATION_LIST}


@app.get("/api/classes")
def get_classes():
    return {
        "classes": [
            {"code": c, "name": DISEASE_FULL_NAMES[c], "risk": DISEASE_RISK[c]}
            for c in CLASS_NAMES
        ]
    }


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)