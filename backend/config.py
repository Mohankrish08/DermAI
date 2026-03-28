# ===============================
# Federated Multimodal Skin Lesion Classification
# Configuration File
# ===============================

import torch

# ===============================
# DATA SETTINGS
# ===============================

NUM_CLASSES = 7
TEST_SPLIT = 0.2
RANDOM_SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32

# ===============================
# FEDERATED LEARNING SETTINGS
# ===============================

NUM_CLIENTS = 2
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.0001
USE_SECURE_AGGREGATION = True

# ===============================
# DIFFERENTIAL PRIVACY SETTINGS
# ===============================

USE_DP = False
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0
DELTA = 1e-5

# ===============================
# DEVICE SETTINGS
# ===============================

DEVICE = torch.device("cpu")

# ===============================
# FILE PATHS
# ===============================

METADATA_PATH = "metadata_balanced.csv"

# Use BOTH image folders
IMAGE_FOLDER = [
    "HAM10000_images",
    "HAM10000_balanced"
]

CENTRALIZED_MODEL_PATH = "centralized_model_Improved.pth"
SECURE_MODE = False   # True only for production
RESULTS_DIR = "results"