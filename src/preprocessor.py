"""
preprocessor.py

UAH-DriveSet preprocessing module.

This module is responsible for reading raw sensor files.
The preprocessing pipeline (merge, cleaning, parquet export)
will be implemented step by step.
"""

from pathlib import Path

import pandas as pd
from config import FEATURE_COLUMNS

# ============================================================
# Column Names
# ============================================================

ACC_COLUMNS = [
    "timestamp",
    "active",
    "acc_x",
    "acc_y",
    "acc_z",
    "acc_x_kf",
    "acc_y_kf",
    "acc_z_kf",
    "roll",
    "pitch",
    "yaw",
]

GPS_COLUMNS = [
    "timestamp",
    "speed",
    "latitude",
    "longitude",
    "altitude",
    "gps_quality",
    "satellites",
    "heading",
    "extra_1",
    "extra_2",
    "extra_3",
    "extra_4",
]

LANE_COLUMNS = [
    "time",
    "lane_offset",
    "phi",
    "road_width",
    "lane_state",
]

VEHICLE_COLUMNS = [
    "time",
    "front_distance",
    "relative_speed",
    "vehicle_state",
    "confidence",
]


# ============================================================
# File Readers
# ============================================================

def load_accelerometer(acc_path: Path) -> pd.DataFrame:
    """Load RAW_ACCELEROMETERS.txt"""

    return pd.read_csv(
        acc_path,
        sep=r"\s+",
        header=None,
        names=ACC_COLUMNS,
    )


def load_gps(gps_path: Path) -> pd.DataFrame:
    """Load RAW_GPS.txt"""

    return pd.read_csv(
        gps_path,
        sep=r"\s+",
        header=None,
        names=GPS_COLUMNS,
    )


def load_lane_detection(lane_path: Path) -> pd.DataFrame:
    """Load PROC_LANE_DETECTION.txt"""

    return pd.read_csv(
        lane_path,
        sep=r"\s+",
        header=None,
        names=LANE_COLUMNS,
    )


def load_vehicle_detection(vehicle_path: Path) -> pd.DataFrame:
    """Load PROC_VEHICLE_DETECTION.txt"""

    return pd.read_csv(
        vehicle_path,
        sep=r"\s+",
        header=None,
        names=VEHICLE_COLUMNS,
    )
    
    
def get_trip_files(trip_path: Path) -> dict:
    """
    Return all raw sensor file paths for a trip.
    """

    return {
        "accelerometer": trip_path / "RAW_ACCELEROMETERS.txt",
        "gps": trip_path / "RAW_GPS.txt",
        "lane": trip_path / "PROC_LANE_DETECTION.txt",
        "vehicle": trip_path / "PROC_VEHICLE_DETECTION.txt",
    }

def load_trip(trip_path: Path) -> dict:
    """
    Load all raw sensor files belonging to a single trip.
    """

    files = get_trip_files(trip_path)

    return {
        "accelerometer": load_accelerometer(files["accelerometer"]),
        "gps": load_gps(files["gps"]),
        "lane": load_lane_detection(files["lane"]),
        "vehicle": load_vehicle_detection(files["vehicle"]),
    }
    
def merge_trip(trip_path: Path) -> pd.DataFrame:
    """
    Load and synchronize all sensor streams for a single trip.

    Parameters
    ----------
    trip_path : Path
        Path to a trip folder.

    Returns
    -------
    pd.DataFrame
        A synchronized dataframe using accelerometer timestamps
        as the reference timeline.
    """

    trip = load_trip(trip_path)

    acc_df = trip["accelerometer"].sort_values("timestamp")
    gps_df = trip["gps"].sort_values("timestamp")
    lane_df = trip["lane"].sort_values("time")
    vehicle_df = trip["vehicle"].sort_values("time")

    # --------------------------------------------------------
    # Merge Accelerometer + GPS
    # --------------------------------------------------------
    merged = pd.merge_asof(
        acc_df,
        gps_df,
        on="timestamp",
        direction="nearest",
    )

    # --------------------------------------------------------
    # Merge Lane Detection
    # --------------------------------------------------------
    merged = pd.merge_asof(
        merged,
        lane_df,
        left_on="timestamp",
        right_on="time",
        direction="nearest",
    )

    # --------------------------------------------------------
    # Merge Vehicle Detection
    # --------------------------------------------------------
    merged = pd.merge_asof(
        merged,
        vehicle_df,
        left_on="timestamp",
        right_on="time",
        direction="nearest",
    )

    return merged

def clean_trip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the features required by the baseline LSTM model.
    """

    cleaned_df = df[FEATURE_COLUMNS].copy()

    return cleaned_df
