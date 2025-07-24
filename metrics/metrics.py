"""
Fichier pour implémenter les métriques permettant d'évaluer la qualité des différents nuages de points par leur proximité avec les nuages de points réels.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
import os
import trimesh
from utils import trim_to_pointcloud
from point_cloud import PointCloud


class Distance:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _normalize(self, pc: PointCloud):
        """
        Ramène le nuage dans le cube [0, 1]^3
        """
        pc.normalize()

    def _prepare(self, pc_1: PointCloud, pc_2: PointCloud):
        """
        Applique la normalisation si nécessaire
        """
        if self.normalize:
            self._normalize(pc_1)
            self._normalize(pc_2)

    def compute_cost_matrix(self, pc_1: PointCloud, pc_2: PointCloud, flatten=False):
        # Matrice de coût
        C = cdist(np.asarray(pc_1.points), np.asarray(pc_2.points), metric="euclidean")
        return C.flatten() if flatten else C

    def earth_mover(self, pc_1: PointCloud, pc_2: PointCloud):
        """
        Fonction pour calculer la distance de Earth Mover entre deux nuages de points.
        """
        self._prepare(pc_1, pc_2)
        # Poids uniformes
        n, m = pc_1.size, pc_2.size
        u = np.full(n, 1.0 / n)
        v = np.full(m, 1.0 / m)
        # Matrice de coût
        C = self.compute_cost_matrix(pc_1, pc_2, flatten=True)
        # Contraintes d'égalités:
        # - pour chaque i: sum_j T_{ij} = u_i → total n contraintes
        # - pour chaque j: sum_i T_{ij} = v_j → total m contraintes
        pc_1_eq = []
        pc_2_eq = []
        # Contraintes lignes (sommation sur les j)
        for i in range(n):
            row = np.zeros(n * m)
            row[i * m : (i + 1) * m] = 1
            pc_1_eq.append(row)
            pc_2_eq.append(u[i])

        # Contraintes colonnes (sommation sur les i)
        for j in range(m):
            col = np.zeros(n * m)
            col[j::m] = 1
            pc_1_eq.append(col)
            pc_2_eq.append(v[j])

        pc_1_eq = np.array(pc_1_eq)
        pc_2_eq = np.array(pc_2_eq)

        bounds = [(0, None) for _ in range(n * m)]

        result = linprog(c=C, A_eq=pc_1_eq, b_eq=pc_2_eq, bounds=bounds, method="highs")

        if result.success:
            emd = result.fun
            return emd
        else:
            raise ValueError("Linear program failed:", result.message)

    def sinkhorn(
        self,
        pc_1: PointCloud,
        pc_2: PointCloud,
        lambda_reg=10.0,
        max_iter=1000,
        tol=1e-5,
    ):
        """
        Distance de Sinkhorn (comme Earth Mover avec un terme de régularisation entropique)
        """
        self._prepare(pc_1, pc_2)
        n, m = pc_1.size, pc_2.size
        # Matrice des coûts : distances euclidiennes
        C = self.compute_cost_matrix(pc_1, pc_2)

        # Matrice K = exp(-lambda * M)
        K = np.exp(-lambda_reg * C)

        # Initialisation
        u = np.ones(n)
        v = np.ones(m)
        # Poids uniformes
        r = np.full(n, 1.0 / n)
        c = np.full(m, 1.0 / m)

        # Boucle principale de Sinkhorn
        for _ in range(max_iter):
            u_prev = u.copy()
            u = r / (K @ v)
            v = c / (K.T @ u)

            if np.linalg.norm(u - u_prev, 1) < tol:
                break

        # Plan de transport final
        P = np.outer(u, v) * K

        # Distance = coût total
        sinkhorn_distance = np.sum(P * C)
        return sinkhorn_distance
        # TODO regarder Python Optimal Transport et travailler en log-domain pour une version plus stable

    def schamfer(self, pc_1: PointCloud, pc_2: PointCloud):
        """
        Distance de Schamfer
        """
        self._prepare(pc_1, pc_2)
        C = self.compute_cost_matrix(pc_1, pc_2)
        n, m = pc_1.size, pc_2.size
        C12 = np.sum(np.min(C, axis=1)) / n
        C21 = np.sum(np.min(C, axis=0)) / m
        return C12 + C21


def compute_distances_to_reference(
    reference_pc_link, dataset, metric="earth_mover", normalize=True
):
    mesh_scene = trimesh.load(reference_pc_link, force="scene")
    ref_point_cloud = np.array(trim_to_pointcloud(mesh_scene))
    distance = Distance(normalize=normalize)
    method = getattr(distance, metric)

    distances = []
    for file in os.listdir(dataset):
        if file.endswith(".npy"):
            point_cloud = np.load(os.path.join(dataset, file))
            distances.append(method(point_cloud, ref_point_cloud))
    distances = np.array(distances)
    return distances


def MSE_MAE(distances):
    mse = np.mean(distances**2)
    mae = np.mean(np.abs(distances))
    return mse, mae


def variability(distances):
    return np.mean((distances - np.mean(distances)) ** 2)
