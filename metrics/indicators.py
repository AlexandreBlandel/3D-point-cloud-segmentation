import numpy as np
from scipy.spatial import KDTree
from abc import abstractmethod


class Indicator:
    def __init__(self, point_cloud: np.ndarray):
        self.point_cloud = point_cloud

    @abstractmethod
    def compute(self):
        raise ValueError(
            ("La méthode compute() doit être implémentée dans la sous-classe.")
        )


class NearestNeighborDistanceIndicator(Indicator):
    def compute(self):
        if self.point_cloud.shape[0] < 2:
            return np.array([])
        tree = KDTree(self.point_cloud)
        dists, _ = tree.query(self.point_cloud, k=2)
        return dists[:, 1]


class DistanceToYAxisIndicator(Indicator):
    def compute(self):
        return np.linalg.norm(self.point_cloud[:, [0, 2]], axis=1)


class YCoordinateDistributionIndicator(Indicator):
    def compute(self):
        return self.point_cloud[:, 1]
