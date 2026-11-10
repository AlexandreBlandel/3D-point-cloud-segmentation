"""
Fichier pour implémenter les métriques permettant d'évaluer la qualité des différents nuages de points par leur proximité avec les nuages de points réels.
"""

import numpy as np
from point_cloud import PointCloud
from pointclouds import PointClouds
from abc import ABC, abstractmethod
from metrics.histogram_distance import HistogramDistances
from metrics.indicators import Indicator
import ot


class PointCloudDistances(ABC):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _normalize(self, pcs: PointClouds):
        """
        Ramène le nuage dans le cube [0, 1]^3
        """
        return(pcs.normalize())

    def _prepare(self, pcs_1: PointClouds, pcs_2: PointClouds):
        """
        Applique la normalisation si nécessaire
        """
        if self.normalize:
            pcs_1=self._normalize(pcs_1)
            pcs_2=self._normalize(pcs_2)
        return pcs_1, pcs_2

    def compute_cost_matrices(self, pcs_1: PointClouds, pcs_2: PointClouds):
        """
        Calcule toutes les matrices de coût (euclidiennes) entre les éléments de pcs_1 et pcs_2.
        
        Retourne :
            - Une matrice 2D (liste de listes) de shape (B1, B2), où chaque élément est
            une matrice de coût de shape (N_i, M_j).
        """
        B1, B2 = len(pcs_1), len(pcs_2)
        cost_matrices = [[None for _ in range(B2)] for _ in range(B1)]

        pcs_1, pcs_2 = self._prepare(pcs_1, pcs_2)

        for i in range(B1):
            X = pcs_1.points[i]
            for j in range(B2):
                Y = pcs_2.points[j]
                cost_matrices[i][j] = ot.dist(X, Y, metric="euclidean")

        return cost_matrices
    
    def mean(self, distances):
        return np.mean(distances)

    def __repr__(self):
        return f"{self.__class__.__name__}(normalize={self.normalize})"

    def __call__(self, pcs_1: PointClouds, pcs_2: PointClouds):
        return self.compute(pcs_1, pcs_2)

    @abstractmethod
    def compute(self, pcs_1:PointClouds, pcs_2:PointClouds):
        pass 

#à rajouter peut-être: downsample les nuages de point trop gros pour la classe fille: max 35_000 points par exemple pour Earth Mover

class EarthMoverDistance(PointCloudDistances):
    def compute(self, pcs_1: PointClouds, pcs_2: PointClouds):
        """
        Calcule la Earth Mover Distance (Wasserstein) entre chaque couple (i,j) de pcs_1 et pcs_2.

        Retourne :
            - Une matrice (liste de listes) de shape (B1, B2), 
            contenant les distances EMD entre pcs_1[i] et pcs_2[j].
        """
        pcs_1, pcs_2 = self._prepare(pcs_1, pcs_2)
        B1, B2 = len(pcs_1), len(pcs_2)

        cost_matrices = self.compute_cost_matrices(pcs_1, pcs_2)
        distances = [[None for _ in range(B2)] for _ in range(B1)]

        for i in range(B1):
            X = pcs_1.points[i]
            n = len(X)
            a = np.full(n, 1.0 / n)

            for j in range(B2):
                Y = pcs_2.points[j]
                m = len(Y)
                b = np.full(m, 1.0 / m)

                M = cost_matrices[i][j]  # (n, m)
                emd_value = ot.emd2(a, b, M, numItermax=1_000_000)
                distances[i][j] = np.sqrt(emd_value)

        return distances

class SinkhornDistance(PointCloudDistances):
    def __init__(self, normalize=True, lambda_reg=10.0, max_iter=1000, tol=1e-5):
        super().__init__(normalize)
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol

    def compute(
        self,
        pcs_1: PointClouds,
        pcs_2: PointClouds,
    ):
        """
        Distance de Sinkhorn (comme Earth Mover avec un terme de régularisation entropique)
        """
        pcs_1, pcs_2 = self._prepare(pcs_1, pcs_2)
        B1, B2 = len(pcs_1), len(pcs_2)

        cost_matrices = self.compute_cost_matrices(pcs_1, pcs_2)
        distances = [[None for _ in range(B2)] for _ in range(B1)]
        for i in range(B1):
            X = np.asarray(pcs_1.points[i])
            n=len(X)
            a = np.full(n, 1.0 / n)
            for j in range(B2):
                Y = np.asarray(pcs_2.points[j])
                m = len(Y)
                b = np.full(m, 1.0 / m)

                M=cost_matrices[i][j]
                sinkhorn_dist = ot.sinkhorn2(a, b, M, reg=self.lambda_reg,
                                            numItermax=self.max_iter, stopThr=self.tol)
                distances[i][j]=np.sqrt(sinkhorn_dist)
        return distances

class ChamferDistance(PointCloudDistances):
    def compute(self, pcs_1: PointClouds, pcs_2: PointClouds):
        """
        Distance de Chamfer
        """
        pcs_1, pcs_2 = self._prepare(pcs_1, pcs_2)
        B1, B2 = len(pcs_1), len(pcs_2)

        cost_matrices = self.compute_cost_matrices(pcs_1, pcs_2)
        distances = [[None for _ in range(B2)] for _ in range(B1)]

        for i in range(B1):
            X = np.asarray(pcs_1.points[i])
            n=len(X)
            for j in range(B2):
                Y = np.asarray(pcs_2.points[j])
                m=len(Y)
                M=cost_matrices[i][j]
                C12 = np.sum(np.min(M, axis=1)) / n
                C21 = np.sum(np.min(M, axis=0)) / m
                distances[i][j]=C12+C21
        return distances

class HistogramIndicator_distance(PointCloudDistances):
    def __init__(self, indicator: Indicator, hist_dist: HistogramDistances, normalize=True, bins=50, range=None, normalize_points=False):
        super().__init__(normalize=normalize)
        self.indicator = indicator
        self.hist_dist = hist_dist
        self.bins = bins
        self.range = range
        self.normalize_points = normalize_points

    def compute(self, pcs_1: PointClouds, pcs_2: PointClouds):
        pcs_1, pcs_2 = self._prepare(pcs_1, pcs_2)

        # Calcul des histogrammes pour chaque batch
        hists_1, _ = pcs_1.compute_histograms(
            indicator=self.indicator,
            bins=self.bins,
            range=self.range,
            normalize_points=self.normalize_points,
            normalize_hist=True
        )

        hists_2, _ = pcs_2.compute_histograms(
            indicator=self.indicator,
            bins=self.bins,
            range=self.range,
            normalize_points=self.normalize_points,
            normalize_hist=True
        )

        return self.hist_dist.compute(hists_1, hists_2)
