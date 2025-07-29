import numpy as np
import random as rd
import trimesh
import pyvista as pv
import pyransame
import math
import os
from abc import ABC, abstractmethod


        # Paramètres de déformation aléatoires pour n corolles (à intégrer dans les paramètres du générateur)
        # sigma = np.random.uniform(0.0005, 0.0015, size=(self.n, 1, 1))
        # trans = np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], size=(self.n, 1, 3))
        # angle = np.random.uniform(0, 2.2, size=(self.n, 1, 1))
        # scale_error = np.random.uniform(-0.05, 0.05, size=(self.n, 1, 1))


class DataAugmentation(ABC):
    def __init__(self, n=1, sigma_range_min=0.0005, sigma_range_max=0.0015, random_sigma=True, use_noise=True, trans_range=0.05, random_trans=True, use_trans=True, angle_range_deg=2.2, random_angle=True, use_rotation=True, scale_error=0.05, random_scale=True, use_scale=True, mean_percent_dropout=24, std_percent_dropout=8, max_percent_dropout=60, use_dropout=False):
        """
        n: nombre d'échantillons qu'on souhaite obtenir par augmentation de données
        le reste correspond aux paramètres d'augmentation
        """
        self.n=n 

        self.sigma_range_min=sigma_range_min
        self.sigma_range_max=sigma_range_max
        self.random_sigma=random_sigma
        self.use_noise=use_noise

        self.trans_range=trans_range
        self.random_trans=random_trans
        self.use_trans=use_trans

        self.angle_range_deg=angle_range_deg
        self.random_angle=random_angle
        self.use_rotation=use_rotation

        self.scale_error=scale_error
        self.random_scale=random_scale
        self.use_scale=use_scale

        self.mean_percent_dropout=mean_percent_dropout
        self.std_percent_dropout=std_percent_dropout
        self.max_percent_dropout=max_percent_dropout
        self.use_dropout=use_dropout

    def ensure_batch(self, points):
        """
        Si `points` est (N, 3), le duplique `n` fois pour obtenir (n, N, 3).
        Sinon, le retourne tel quel.
        """
        if points.ndim == 2:
            return np.tile(points[None, :, :], (self.n, 1, 1))
        elif points.ndim == 3:
            return points
        else:
            raise ValueError("points doit être de forme (N, 3) ou (n, N, 3)")

    def ajout_bruit_gaussien(self, points):
        points = self.ensure_batch(points)
        if self.random_sigma:
            sigma = np.random.uniform(self.sigma_range_min, self.sigma_range_max, size=(points.shape[0], 1, 1))
        else:
            sigma = 0.0
        noise = np.random.normal(0, sigma, size=points.shape)
        points += noise
        return points

    def apply_translation(self, points):
        points = self.ensure_batch(points)
        if self.random_trans:
            trans = np.random.uniform(-self.trans_range, self.trans_range, size=(points.shape[0], 1, 3))
        else:
            trans = np.zeros((points.shape[0], 1, 3))
        points += trans
        return points

    def apply_random_rotation(self, points):
        points = self.ensure_batch(points)

        angles = np.deg2rad(np.random.uniform(-self.angle_range_deg, self.angle_range_deg, size=(points.shape[0], 3)))
        R = np.array([self.rotation_matrix_xyz(ax, ay, az) for ax, ay, az in angles])
        points = np.matmul(points, np.transpose(R, (0, 2, 1)))
        return points

    @staticmethod
    def rotation_matrix_xyz(angle_x, angle_y, angle_z):
        Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def apply_scaling(self, points):
        points = self.ensure_batch(points)
        if self.random_scale:
            scale = np.random.uniform(1 - self.scale_error, 1 + self.scale_error, size=(points.shape[0], 1, 1))
        else:
            scale = 1 + self.scale_error
        points*=scale
        return points

    def deform_tot(self, points):
        if self.use_noise:
            points=self.ajout_bruit_gaussien(points)
        if self.use_trans:
            points=self.apply_translation(points)
        if self.use_rotation:
            points=self.apply_random_rotation(points)
        if self.use_scale:
            points=self.apply_scaling(points)
        if self.use_dropout:
            points=self.dropout(points)
        return (points)


    @staticmethod
    def dropout_points(points, mean_percent=24, std_percent=8, max_percent=60):
        """
        Supprime aléatoirement une proportion de points dans un nuage de points,
        avec un taux de dropout tiré d'une loi normale tronquée.

        Args:
            points (np.ndarray): Tableau (N, 3) de points 3D.
            mean_percent (float): Moyenne de la loi normale du taux de dropout.
            std_percent (float): Écart-type de la loi normale.
            max_percent (float): Taux maximum de dropout (en %).

        Returns:
            np.ndarray: Nuage de points après dropout.
        """
        if points is None:
            return None

        points = np.asarray(points)
        n_points = points.shape[0]
        if n_points == 0:
            return points

        # Taux de dropout aléatoire (en %), tronqué entre 0 et max_percent
        dropout_percent = np.clip(
            np.random.normal(loc=mean_percent, scale=std_percent), 0, max_percent
        )
        dropout_ratio = dropout_percent / 100

        # Nombre de points à garder

        n_keep = int(n_points * (1 - dropout_ratio))
        n_keep = max(n_keep, 1)  # on garde au moins 1 point

        # Sélection aléatoire sans remise
        indices = np.random.choice(n_points, n_keep, replace=False)
        return points[indices]

    def dropout(self, points):
        if points is None:
            return None
        if points.ndim == 3:
            return np.array([self.dropout_points(p, self.mean_percent_dropout, self.std_percent_dropout, self.max_percent_dropout) for p in points], dtype=object)
        else:
            return self.dropout_points(points, self.mean_percent_dropout, self.std_percent_dropout, self.max_percent_dropout)
