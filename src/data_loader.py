"""
data_loader.py

PyTorch Dataset and DataLoader utilities for the
UAH Driver Risk Prediction project.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from preprocessor import merge_trip, clean_trip

from config import (
    SEQUENCE_LENGTH,
    STRIDE,
    FEATURE_COLUMNS,
    LABEL_MAPPING,
)

def create_sequences(
    data: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
):
    """
    Convert a continuous sensor stream into overlapping sequences.

    Parameters
    ----------
    data : np.ndarray
        Shape = (num_samples, num_features)

    Returns
    -------
    np.ndarray
        Shape = (num_sequences, sequence_length, num_features)
    """

    sequences = []

    for start in range(
        0,
        len(data) - sequence_length + 1,
        stride,
    ):

        end = start + sequence_length

        sequences.append(data[start:end])

    return np.array(sequences)

from config import LABEL_MAPPING


def extract_trip_info(trip_name: str) -> dict:
    """
    Extract driver, behaviour and road type from a trip folder name.

    Example
    -------
    20151111123124-25km-D1-NORMAL-MOTORWAY
    """

    parts = trip_name.split("-")

    behaviour = parts[3]

    if behaviour.startswith("NORMAL"):
        behaviour = "NORMAL"

    return {
        "driver": parts[2],
        "behaviour": behaviour,
        "road": parts[4],
        "label": LABEL_MAPPING[behaviour],
    }
    
def get_all_trip_paths(dataset_path: Path):
    """
    Collect all valid trip folders from the UAH-DriveSet dataset.
    """

    trip_paths = []

    for driver_folder in sorted(dataset_path.iterdir()):

        # Sadece D1 ... D6 klasörlerini al
        if (
            not driver_folder.is_dir()
            or not driver_folder.name.startswith("D")
        ):
            continue

        for trip_folder in sorted(driver_folder.iterdir()):

            # Sadece gerçek trip klasörlerini al
            if (
                not trip_folder.is_dir()
                or not trip_folder.name.startswith("20")
            ):
                continue

            required_files = [
                "RAW_ACCELEROMETERS.txt",
                "RAW_GPS.txt",
                "PROC_LANE_DETECTION.txt",
                "PROC_VEHICLE_DETECTION.txt",
            ]

            # Gerçek trip mi kontrol et
            if all((trip_folder / f).exists() for f in required_files):
                trip_paths.append(trip_folder)

    return trip_paths

def build_dataset(dataset_path: Path):
    """
    Build the complete sequence dataset.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    groups : np.ndarray
    """

    X = []
    y = []
    groups = []

    trip_paths = get_all_trip_paths(dataset_path)

    for group_id, trip_path in enumerate(trip_paths):

        info = extract_trip_info(trip_path.name)

        merged_df = merge_trip(trip_path)

        clean_df = clean_trip(merged_df)

        sequences = create_sequences(clean_df.to_numpy())

        X.append(sequences)

        y.extend([info["label"]] * len(sequences))

        groups.extend([group_id] * len(sequences))

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    groups = np.array(groups)

    assert len(X) == len(y)
    assert len(X) == len(groups)

    print(f"Dataset Shape : {X.shape}")
    print(f"Labels        : {y.shape}")
    print(f"Groups        : {groups.shape}")

    return X, y, groups

def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    save_path: Path,
):
    """
    Save processed dataset to disk.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        X=X,
        y=y,
        groups=groups,
    )

    print(f"Dataset saved to: {save_path}")


def load_dataset(save_path: Path):
    """
    Load processed dataset from disk.
    """

    data = np.load(save_path)

    return (
        data["X"],
        data["y"],
        data["groups"],
    )