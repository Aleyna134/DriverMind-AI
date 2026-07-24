"""
config.py

Global configuration file for the UAH Driver Risk Prediction project.

This file contains all project-wide constants used by the
preprocessing pipeline, dataloaders, model training and evaluation.
"""

# ============================================================
# Randomness
# ============================================================

RANDOM_STATE = 42


# ============================================================
# Dataset
# ============================================================

SEQUENCE_LENGTH = 160
STRIDE = 10


# ============================================================
# Training
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 1e-4


# ============================================================
# LSTM Input Features
# ============================================================

FEATURE_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",

    "roll",
    "pitch",
    "yaw",

    "speed",
    "heading",

    "lane_offset",
    "phi",

    "front_distance",
    "relative_speed",

    "vehicle_state",
]


# ============================================================
# Behaviour Labels
# ============================================================

LABEL_MAPPING = {
    "NORMAL": 0,
    "DROWSY": 1,
    "AGGRESSIVE": 2,
}