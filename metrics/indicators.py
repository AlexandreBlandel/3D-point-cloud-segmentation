import numpy as np
from scipy.spatial import KDTree
from point_cloud import PointCloud
from pointclouds import PointClouds
from abc import abstractmethod


class Indicator():
    """
    classe abstraite
    """
    @abstractmethod
    def compute(self, pcs:PointClouds):
        raise NotImplementedError(
            ("compute() doit être implémentée dans la sous-classe.")
        )

    def __call__(self, pcs: PointClouds):
        return self.compute(pcs)



class NearestNeighborDistanceIndicator(Indicator):
    def compute(self, pcs:PointClouds):
        B=len(pcs)
        indicator_values=[None for _ in range(B)]
        for i in range(B):
            pc=pcs[i]
            if pc.points.shape[0] < 2:
                indicator_values[i] = np.array([])
                continue
            tree = KDTree(pc.points)
            value, _ = tree.query(pc.points, k=2)
            indicator_values[i]=value[:, 1]
        return indicator_values


class DistanceToYAxisIndicator(Indicator):
    def compute(self, pcs: PointClouds):
        return [np.linalg.norm(pc.points[:, [0, 2]], axis=1) for pc in pcs]




class YCoordinateDistributionIndicator(Indicator):
    def compute(self, pcs: PointClouds):
        return [pc.points[:, 1] for pc in pcs]

