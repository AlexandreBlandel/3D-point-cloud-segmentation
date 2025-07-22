import numpy as np
from scipy.spatial import KDTree


class Indicators:
    def __init__(self, point_cloud: np.ndarray):
        self.point_cloud = point_cloud

    def compute(self, indicator_name: str):
        """
        Calcule l'indicateur spécifié par son nom.
        """
        if indicator_name == "nearest_distance":
            return self.nearest_neighbor_distance()
        elif indicator_name == "distance_to_y_axis":
            return self.distance_to_y_axis()
        elif indicator_name == "y_distribution":
            return self.y_coordinates()
        else:
            raise ValueError(f"Indicateur inconnu : {indicator_name}")

    def nearest_neighbor_distance(self):
        if self.point_cloud.shape[0] < 2:
            return np.array([])
        tree = KDTree(self.point_cloud)
        dists, _ = tree.query(self.point_cloud, k=2)
        return dists[:, 1]  # distance au plus proche voisin

    def distance_to_y_axis(self):
        return np.linalg.norm(self.point_cloud[:, [0, 2]], axis=1)

    def y_coordinates(self):
        return self.point_cloud[:, 1]

    @staticmethod
    def list_available():
        return ["nearest_distance", "distance_to_y_axis", "y_distribution"]
