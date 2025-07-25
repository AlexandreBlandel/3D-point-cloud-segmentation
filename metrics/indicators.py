import numpy as np
from scipy.spatial import KDTree
from abc import ABC, abstractmethod
from point_cloud import PointCloud


class Indicator(ABC):
    def __init__(self, points: np.ndarray, normalize=True):
        self.points = points
        self.normalize = normalize

    def _get_points(self):
        return (
            PointCloud.normalize_points(self.points) if self.normalize else self.points
        )

    @abstractmethod
    def compute(self):
        raise ValueError(
            ("La méthode compute() doit être implémentée dans la sous-classe.")
        )


class NearestNeighborDistanceIndicator(Indicator):
    def compute(self):
        points = self._get_points()
        if points.shape[0] < 2:
            return np.array([])
        tree = KDTree(points)
        dists, _ = tree.query(points, k=2)
        return dists[:, 1]


class DistanceToYAxisIndicator(Indicator):
    def compute(self):
        points = self._get_points()
        return np.linalg.norm(points[:, [0, 2]], axis=1)


class YCoordinateDistributionIndicator(Indicator):
    def compute(self):
        points = self._get_points()
        return points[:, 1]
