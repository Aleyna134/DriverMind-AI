from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SimulationEngine:

    def __init__(self, project_path):

        self.project_path = Path(project_path)

        processed = self.project_path / "datasets" / "processed"

        # Kullanılacak 7 özellik
        self.simulation_vectors = np.load(
            processed / "simulation_vectors_scaled.npy"
        )

        # StandardScaler
        self.scaler = joblib.load(
            processed / "simulation_scaler.pkl"
        )

        # Gerçek LSTM girişleri
        data = np.load(
            processed / "uah_dataset_window160.npz"
        )

        self.X = data["X"]

        # Nearest Neighbor modeli
        self.nn = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean"
        )

        self.nn.fit(self.simulation_vectors)

    def find_window(
        self,
        speed,
        heading,
        lane_offset,
        front_distance,
        relative_speed,
        vehicle_state,
        acc_z,
    ):

        return self.find_windows(
            speed=speed,
            heading=heading,
            lane_offset=lane_offset,
            front_distance=front_distance,
            relative_speed=relative_speed,
            vehicle_state=vehicle_state,
            acc_z=acc_z,
            k=1,
        )[0]

    def find_windows(
        self,
        speed,
        heading,
        lane_offset,
        front_distance,
        relative_speed,
        vehicle_state,
        acc_z,
        k=20,
    ):
        """
        Return the k nearest real driving windows to the given
        control values. Averaging the model over several
        neighbours (instead of a single one) smooths out the
        noise that comes from any one recorded trip.
        """

        user_vector = np.array([[
            speed,
            heading,
            lane_offset,
            front_distance,
            relative_speed,
            vehicle_state,
            acc_z
        ]])

        user_vector = self.scaler.transform(user_vector)

        _, indices = self.nn.kneighbors(user_vector, n_neighbors=k)

        return self.X[indices[0]]