import numpy as np
from scipy.spatial import KDTree
from point_cloud import PointCloud
from abc import abstractmethod


class Indicator():
    """
    classe abstraite
    """
    @abstractmethod
    def compute(self, pc:PointCloud):
        raise NotImplementedError(
            ("compute() doit être implémentée dans la sous-classe.")
        )

    def __call__(self, pc: PointCloud):
        return self.compute(pc)



class NearestNeighborDistanceIndicator(Indicator):
    def compute(self, pc:PointCloud):
        if pc.points.shape[0] < 2:
            return np.array([])
        tree = KDTree(pc.points)
        dists, _ = tree.query(pc.points, k=2)
        return dists[:, 1]


class DistanceToYAxisIndicator(Indicator):
    def compute(self, pc: PointCloud):
        return np.linalg.norm(pc.points[:, [0, 2]], axis=1)


class YCoordinateDistributionIndicator(Indicator):
    def compute(self, pc: PointCloud):
        return pc.points[:, 1]
