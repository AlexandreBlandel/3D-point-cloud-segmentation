"""
Fichier pour implémenter les métriques permettant d'évaluer la qualité des différents nuages de points par leur proximité avec les nuages de points réels.
"""

import numpy as np
from point_cloud import PointCloud
from abc import ABC, abstractmethod
import ot


class Distance(ABC):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _normalize(self, pc: PointCloud):
        """
        Ramène le nuage dans le cube [0, 1]^3
        """
        return(pc.normalize())

    def _prepare(self, pc_1: PointCloud, pc_2: PointCloud):
        """
        Applique la normalisation si nécessaire
        """
        if self.normalize:
            pc_1=self._normalize(pc_1)
            pc_2=self._normalize(pc_2)
        return pc_1, pc_2

    def compute_cost_matrix(self, pc_1: PointCloud, pc_2: PointCloud):
        """
        Calcule la matrice de coût (euclidienne) entre deux nuages de points avec POT
        """
        X = np.asarray(pc_1.points)
        Y = np.asarray(pc_2.points)
        return ot.dist(X, Y, metric="euclidean")

    def __repr__(self):
        return f"{self.__class__.__name__}(normalize={self.normalize})"

    def __call__(self, pc_1: PointCloud, pc_2: PointCloud):
        return self.compute(pc_1, pc_2)

    @abstractmethod
    def compute(self, pc_1:PointCloud, pc_2:PointCloud):
        pass 

class EarthMoverDistance(Distance):
    def compute(self, pc_1: PointCloud, pc_2: PointCloud):
        """
        Distance Earth Mover (Wasserstein) avec POT.
        """
        pc_1, pc_2 = self._prepare(pc_1, pc_2)

        X = np.asarray(pc_1.points)  # (n, 3)
        Y = np.asarray(pc_2.points)  # (m, 3)

        n, m = len(X), len(Y)

        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)

        M = self.compute_cost_matrix(pc_1, pc_2) 

        emd_value = ot.emd2(a, b, M, numItermax=1_000_000)

        return np.sqrt(emd_value)

class SinkhornDistance(Distance):
    def __init__(self, normalize=True, lambda_reg=10.0, max_iter=1000, tol=1e-5):
        super().__init__(normalize)
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol

    def compute(
        self,
        pc_1: PointCloud,
        pc_2: PointCloud,
    ):
        """
        Distance de Sinkhorn (comme Earth Mover avec un terme de régularisation entropique)
        """
        pc_1, pc_2 = self._prepare(pc_1, pc_2)
        X = np.asarray(pc_1.points)
        Y = np.asarray(pc_2.points)

        n, m = len(X), len(Y)
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)

        M = self.compute_cost_matrix(pc_1, pc_2)
        sinkhorn_dist = ot.sinkhorn2(a, b, M, reg=self.lambda_reg,
                                      numItermax=self.max_iter, stopThr=self.tol)

        return np.sqrt(sinkhorn_dist)

class SchamferDistance(Distance):
    def compute(self, pc_1: PointCloud, pc_2: PointCloud):
        """
        Distance de Schamfer
        """
        pc_1, pc_2=self._prepare(pc_1, pc_2)
        M = self.compute_cost_matrix(pc_1, pc_2)
        n, m = pc_1.size, pc_2.size
        C12 = np.sum(np.min(M, axis=1)) / n
        C21 = np.sum(np.min(M, axis=0)) / m
        return C12 + C21
