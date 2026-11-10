import numpy as np
import random as rd
import trimesh
import pyvista as pv
import pyransame
import math
import os
from abc import ABC, abstractmethod
from point_cloud import PointCloud
from batch_point_cloud import BatchPointCloud


# Paramètres de déformation aléatoires pour n corolles (à intégrer dans les paramètres du générateur)
# sigma = np.random.uniform(0.0005, 0.0015, size=(self.n, 1, 1))
# trans = np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], size=(self.n, 1, 3))
# angle = np.random.uniform(0, 2.2, size=(self.n, 1, 1))
# scale_error = np.random.uniform(-0.05, 0.05, size=(self.n, 1, 1))


#utilisation:
#    base_points = np.broadcast_to(base_points, (n, M, 3)).copy()
#    base_points_deformed = deformation_tot(
#        base_points, sigma, trans, angle, scale_error
#    )

class DataAugmenter:
    def __init__(
        self,
        sigma_min, sigma_max,
        trans,
        angle_range_deg,
        scale,
        min_dropout_ratio, max_dropout_ratio,
        max_points=25_000, dropout_borne=True,
        rd_sigma=True, rd_trans=True, rd_angle=True, rd_scale=True, rd_dropout=True
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.trans = trans
        self.angle_range_deg = angle_range_deg
        self.scale = scale
        self.min_dropout_ratio = min_dropout_ratio
        self.max_dropout_ratio = max_dropout_ratio
        self.max_points = max_points
        self.dropout_borne = dropout_borne

        self.rd_sigma = rd_sigma
        self.rd_trans = rd_trans
        self.rd_angle = rd_angle
        self.rd_scale = rd_scale
        self.rd_dropout = rd_dropout

    def ajout_bruit_gaussien(self, pcs):
        for i in range(len(pcs)):
            n = pcs.points[i].shape[0]
            sigma = np.random.uniform(self.sigma_min, self.sigma_max) if self.rd_sigma else self.sigma_min
            noise = np.random.normal(0, sigma, size=(n, 3))
            pcs.points[i] += noise
        return pcs

    def apply_translation(self, pcs):
        for i in range(len(pcs)):
            translation = np.random.uniform(-self.trans, self.trans, size=(1, 3)) if self.rd_trans else np.zeros((1, 3))
            pcs.points[i] += translation
        return pcs

    def apply_rotation(self, pcs, axis="x", degrees=None):
        """
        Applique une rotation autour de l'axe spécifié à chaque nuage de points du batch.

        Paramètres :
            - pcs : un objet PointClouds
            - axis : 'x', 'y' ou 'z'
            - degrees : si None, tiré aléatoirement selon angle_range_deg
        """
        deg = np.random.uniform(-self.angle_range_deg, self.angle_range_deg) if degrees is None else degrees
        radians = np.deg2rad(deg)
        if axis == "x":
            R = self.rotation_matrix_xyz(radians, 0, 0)
        elif axis == "y":
            R = self.rotation_matrix_xyz(0, radians, 0)
        elif axis == "z":
            R = self.rotation_matrix_xyz(0, 0, radians)
        else:
            raise ValueError("Axis must be 'x', 'y' or 'z'")
        for i in range(len(pcs)):
            pcs.points[i] = pcs.points[i] @ R.T
        return pcs


    def apply_random_rotation(self, pcs):
        for i in range(len(pcs)):
            angles = np.deg2rad(np.random.uniform(-self.angle_range_deg, self.angle_range_deg, size=3))
            R = self.rotation_matrix_xyz(*angles)
            pcs.points[i] = pcs.points[i] @ R.T
        return pcs

    def apply_scaling(self, pcs):
        for i in range(len(pcs)):
            s = np.random.uniform(1 - self.scale, 1 + self.scale) if self.rd_scale else 1 + self.scale
            pcs.points[i] *= s
        return pcs

    def apply_dropout(self, pcs):
        for i in range(len(pcs)):
            n = pcs.points[i].shape[0]
            ratio = np.random.uniform(self.min_dropout_ratio, self.max_dropout_ratio) if self.rd_dropout else self.min_dropout_ratio
            n_keep = max(int((1 - ratio) * n), 1)
            if self.dropout_borne:
                n_keep = min(n_keep, self.max_points)
            idx = np.random.choice(n, n_keep, replace=False)
            pcs.points[i] = pcs.points[i][idx]
            pcs.colors[i] = pcs.colors[i][idx]
            if pcs.labels is not None:
                pcs.labels[i] = pcs.labels[i][idx]
        return pcs

    @staticmethod
    def rotation_matrix_xyz(angle_x, angle_y, angle_z):
        Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def __repr__(self):
        return f"{self.__class__.__name__}(" + ", ".join([
            f"sigma=({self.sigma_min}, {self.sigma_max})",
            f"trans={self.trans}",
            f"angle_range_deg={self.angle_range_deg}",
            f"scale={self.scale}",
            f"dropout=({self.min_dropout_ratio}, {self.max_dropout_ratio})"
        ]) + ")"


"""
class DataAugmenter(ABC):
    def __init__(
        self,
        sigma_min, sigma_max,
        trans, 
        angle_range_deg,  
        scale, 
        min_dropout_ratio, max_dropout_ratio, max_points=25_000, dropout_borne=True,
        rd_sigma=True, rd_trans=True, rd_angle=True, rd_scale=True, rd_dropout=True
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rd_sigma = rd_sigma

        self.trans = trans
        self.rd_trans = rd_trans

        self.angle_range_deg = angle_range_deg
        self.rd_angle = rd_angle

        self.scale = scale
        self.rd_scale = rd_scale

        self.min_dropout_ratio = min_dropout_ratio
        self.max_dropout_ratio = max_dropout_ratio
        self.dropout_borne=dropout_borne
        self.max_points = max_points
        self.rd_dropout=rd_dropout

    @abstractmethod
    def ajout_bruit_gaussien(self, pc):
        pass
    
    @abstractmethod
    def apply_translation(self, pc):
        pass

    @staticmethod
    def rotation_matrix_xyz(angle_x, angle_y, angle_z):
        Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def apply_rotation(self, pc, axis, degrees):
        radians = np.deg2rad(degrees)
        if axis == "x":
            R = self.rotation_matrix_xyz(radians, 0, 0)
        elif axis == "y":
            R = self.rotation_matrix_xyz(0, radians, 0)
        elif axis == "z":
            R = self.rotation_matrix_xyz(0, 0, radians)
        else:
            raise ValueError("Axis must be 'x', 'y' or 'z'")

        if len(pc.points.shape) == 2: 
            pc.points = pc.points @ R.T
        elif len(pc.points.shape) == 3:  
            B, N, _ = pc.points.shape
            pc.points = (pc.points - 0) @ R.T + 0
        return pc

    def __repr__(self):
        return f"{self.__class__.__name__}(" + ", ".join([
            f"sigma=({self.sigma_min}, {self.sigma_max})",
            f"trans={self.trans}",
            f"angle_range_deg={self.angle_range_deg}",
            f"scale={self.scale}",
            f"dropout=({self.min_dropout_ratio}, {self.max_dropout_ratio})"
        ]) + ")"

    
    @abstractmethod
    def apply_random_rotation(self, pc):
        pass

    @abstractmethod
    def apply_scaling(self, pc):
        pass

    @abstractmethod
    def apply_dropout(self, pc):
        pass
    


class PCDataAugmenter(DataAugmenter):
    def __init__(
        self,
        sigma_min, sigma_max, 
        trans, 
        angle_range_deg,
        scale, 
        min_dropout_ratio, max_dropout_ratio, max_points=25_000, dropout_borne=True, 
        rd_sigma=True, rd_trans=True, rd_angle=True, rd_scale=True, rd_dropout=True
    ):
        super().__init__(
            sigma_min, sigma_max, 
            trans, 
            angle_range_deg, 
            scale, 
            min_dropout_ratio, max_dropout_ratio, max_points, dropout_borne,
            rd_sigma, rd_trans, rd_angle, rd_scale, rd_dropout
            )

    def ajout_bruit_gaussien(self, pc):
        if self.rd_sigma:
            sigma = np.random.uniform(self.sigma_min, self.sigma_max, size=(pc.points.shape[0], 1))
        else:
            sigma = np.full((pc.points.shape[0], 1), self.sigma_min)
        noise = np.random.normal(0, sigma, size=pc.points.shape)
        pc.points += noise
        return pc

    def apply_translation(self, pc):
        if self.rd_trans:
            translation = np.random.uniform(-self.trans, self.trans, size=(1, 3))
        else:
            translation = np.zeros((1, 3))
        pc.points += translation
        return pc

    def apply_random_rotation(self, pc):
        angles = np.deg2rad(np.random.uniform(-self.angle_range_deg, self.angle_range_deg, size=3))
        R = self.rotation_matrix_xyz(*angles)
        pc.points = pc.points @ R.T
        return pc

    def apply_scaling(self, pc):
        if self.rd_scale:
            scale = np.random.uniform(1 - self.scale, 1 + self.scale)
        else:
            scale = 1 + self.scale
        pc.points *= scale
        return pc

    def apply_dropout(self, pc):
        n_points = pc.points.shape[0]
        ratio = np.random.uniform(self.min_dropout_ratio, self.max_dropout_ratio) if self.rd_dropout else self.min_dropout_ratio
        n_keep = max(int((1 - ratio) * n_points), 1)
        if n_keep > self.max_points and self.dropout_borne:
            print(f"number of points limited to {self.max_points} instead of {n_keep} for {pc.name}")
            n_keep=self.max_points
        idx = np.random.choice(n_points, n_keep, replace=False)
        pc.points = pc.points[idx]
        pc.colors = pc.colors[idx]
        if pc.labels is not None:
            pc.labels = pc.labels[idx]
        return pc

class BatchPCDataAugmenter(DataAugmenter):
    def __init__(
        self,
        sigma_min, sigma_max, 
        trans, 
        angle_range_deg,
        scale, 
        min_dropout_ratio, max_dropout_ratio, max_points=25_000, dropout_borne=True,
        rd_sigma=True, rd_trans=True, rd_angle=True, rd_scale=True, rd_dropout=True
    ):
        super().__init__(
            sigma_min, sigma_max, 
            trans, 
            angle_range_deg, 
            scale, 
            min_dropout_ratio, max_dropout_ratio, max_points, dropout_borne,
            rd_sigma, rd_trans, rd_angle, rd_scale, rd_dropout
            )

    def ajout_bruit_gaussien(self, pcs):
        B, N, _ = pcs.points.shape
        if self.rd_sigma:
            sigma = np.random.uniform(self.sigma_min, self.sigma_max, size=(B, N, 1))
        else:
            sigma = np.full((B, N, 1), self.sigma_min)
        noise = np.random.normal(0, sigma, size=pcs.points.shape)
        pcs.points += noise
        return pcs

    def apply_translation(self, pcs):
        B, N, _ = pcs.points.shape
        if self.rd_trans:
            translation = np.random.uniform(-self.trans, self.trans, size=(B, 1, 3))
        else:
            translation = np.zeros((B, 1, 3))
        pcs.points += translation
        return pcs

    def apply_random_rotation(self, pcs):
        B, _, _ = pcs.points.shape
        angles = np.deg2rad(np.random.uniform(-self.angle_range_deg, self.angle_range_deg, size=(B, 3)))
        points = pcs.points
        rotated = np.empty_like(points)
        for i in range(B):
            R = self.rotation_matrix_xyz(*angles[i])
            rotated[i] = points[i] @ R.T
        pcs.points = rotated
        return pcs

    def apply_scaling(self, pcs):
        B, N, _ = pcs.points.shape
        if self.rd_scale:
            scale = np.random.uniform(1 - self.scale, 1 + self.scale, size=(B, 1, 1))
        else:
            scale = np.full((B, 1, 1), 1 + self.scale)
        pcs.points *= scale
        return pcs

    def apply_dropout(self, pcs):
        B, N, _ = pcs.points.shape
        new_points = []
        new_colors = []
        new_labels = [] if pcs.labels is not None else None

        for i in range(B):
            ratio = np.random.uniform(self.min_dropout_ratio, self.max_dropout_ratio) if self.rd_dropout else self.min_dropout_ratio
            n_keep = max(int((1 - ratio) * N), 1)
            if self.dropout_borne:
                n_keep = min(n_keep, self.max_points)
            idx = np.random.choice(N, n_keep, replace=False)

            new_points.append(pcs.points[i, idx])
            new_colors.append(pcs.colors[i, idx])
            if pcs.labels is not None:
                new_labels.append(pcs.labels[i, idx])

        pcs.points = np.array(new_points)
        pcs.colors = np.array(new_colors)
        pcs.labels = np.array(new_labels) if new_labels is not None else None
        return pcs



def ajout_bruit_gaussien(pc, sigma_min, sigma_max, rd_sigma: bool = True):
    
    Applies gaussian noise on the points
    
    if isinstance(pc, PointCloud):
        n_points = pc.points.shape[0]
        if rd_sigma:
            sigma = np.random.uniform(sigma_min, sigma_max, size=(n_points, 1))
        else:
            sigma = np.full((n_points, 1), sigma_min)
        noise = np.random.normal(0, sigma, size=pc.points.shape)
        pc.points += noise
        return pc

    elif isinstance(pc, BatchPointCloud):
        B, N, _ = pc.points.shape
        if rd_sigma:
            sigma = np.random.uniform(sigma_min, sigma_max, size=(B, N, 1))
        else:
            sigma = np.full((B, N, 1), sigma_min)
        noise = np.random.normal(0, sigma, size=pc.points.shape)
        pc.points += noise
        return pc

    else:
        raise TypeError("Input must be a PointCloud or BatchPointCloud")

def apply_translation(pc, trans: float, rd_trans: bool = True):
    
    Applies a random translation on each PointCloud of the batch in [-trans, trans]
    On each point cloud, the translation is uniform
    
    if isinstance(pc, PointCloud):
        if rd_trans:
            translation = np.random.uniform(-trans, trans, size=(1, 3))
        else:
            translation = np.zeros((1, 3))
        pc.points += translation
        return pc

    elif isinstance(pc, BatchPointCloud):
        B, N, _ = pc.points.shape
        if rd_trans:
            translation = np.random.uniform(-trans, trans, size=(B, 1, 3))
        else:
            translation = np.zeros((B, 1, 3))
        pc.points += translation  
        return pc

    else:
        raise TypeError("Input must be a PointCloud or BatchPointCloud")

def rotation_matrix_xyz(angle_x, angle_y, angle_z):
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def apply_rotation(pc, axis, degrees):
    radians = np.deg2rad(degrees)
    if axis=="x":
        R=rotation_matrix_xyz(radians, 0, 0)
    elif axis=="y":
        R=rotation_matrix_xyz(0, radians, 0)
    elif axis=="z":
        R=rotation_matrix_xyz(0, 0, radians)
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")
    
    if isinstance(pc, PointCloud):
        center = np.zeros((1, 3))
        pc.points = (pc.points - center) @ R.T + center
        return pc

    elif isinstance(pc, BatchPointCloud):
        B, N, _ = pc.points.shape
        center = np.zeros((1, 1, 3))
        pc.points = (pc.points - center) @ R.T + center
        return pc
    else:
        raise TypeError("Input must be a PointCloud or BatchPointCloud")

def apply_random_rotation(pc, angle_range_deg: float):
    
    Applies a random rotation around all 3 axes with angles in [-angle_range_deg, angle_range_deg].
    
    if isinstance(pc, PointCloud):
        angles = np.deg2rad(np.random.uniform(-angle_range_deg, angle_range_deg, size=3))
        R = rotation_matrix_xyz(*angles)
        pc.points = pc.points @ R.T
        return pc

    elif isinstance(pc, BatchPointCloud):
        B, _, _ = pc.points.shape
        angles = np.deg2rad(np.random.uniform(-angle_range_deg, angle_range_deg, size=(B, 3)))
        pcs = pc.points

        rotated = np.empty_like(pcs)
        for i in range(B):
            R = rotation_matrix_xyz(*angles[i])
            rotated[i] = pcs[i] @ R.T
        pc.points = rotated
        return pc

    else:
        raise TypeError("Input must be a PointCloud or BatchPointCloud")

def apply_scaling(pc, scale: float, rd_scale: bool = True):
    
    Applies scaling to the point cloud(s).
    
    if isinstance(pc, PointCloud):
        if rd_scale:
            scale = np.random.uniform(1 - scale, 1 + scale)
        else:
            scale = 1 + scale
        pc.points *= scale
        return pc

    elif isinstance(pc, BatchPointCloud):
        B, N, _ = pc.points.shape
        if rd_scale:
            scale = np.random.uniform(1 - scale, 1 + scale, size=(B, 1, 1))
        else:
            scale = np.full((B, 1, 1), 1 + scale)
        pc.points *= scale
        return pc

    else:
        raise TypeError("Input must be a PointCloud or BatchPointCloud")

def apply_dropout(pc, max_dropout_ratio: float, min_dropout_ratio: float, rd_ratio: bool = True, max_points: int = 25_000):
    
    Randomly drops points from the point cloud(s) to simulate partial observations.
    If input is a PointCloud, returns a subsampled PointCloud.
    If input is a BatchPointCloud, applies dropout independently to each sample.
    
    if isinstance(pc, PointCloud):
        n_points = pc.points.shape[0]
        ratio = np.random.uniform(min_dropout_ratio, max_dropout_ratio) if rd_ratio else min_dropout_ratio
        n_keep = max(int((1 - ratio) * n_points), 1)
        n_keep = min(n_keep, max_points)
        idx = np.random.choice(n_points, n_keep, replace=False)
        pc.points = pc.points[idx]
        pc.colors = pc.colors[idx]
        if pc.labels is not None:
            pc.labels = pc.labels[idx]
        return pc

    elif isinstance(pc, BatchPointCloud):
        B, N, _ = pc.points.shape
        new_points = []
        new_colors = []
        new_labels = [] if pc.labels is not None else None

        for i in range(B):
            n_points = N
            ratio = np.random.uniform(min_dropout_ratio, max_dropout_ratio) if rd_ratio else min_dropout_ratio
            n_keep = max(int((1 - ratio) * n_points), 1)
            n_keep = min(n_keep, max_points)
            idx = np.random.choice(n_points, n_keep, replace=False)

            new_points.append(pc.points[i, idx])
            new_colors.append(pc.colors[i, idx])
            if pc.labels is not None:
                new_labels.append(pc.labels[i, idx])

        # Attention : les nuages peuvent avoir des tailles différentes => plus un vrai batch régulier
        pc.points = new_points
        pc.colors = new_colors
        pc.labels = new_labels if new_labels else None
        return pc

    else:
        raise TypeError("Input must be a PointCloud or BatchPointCloud")
"""




"""
def ajout_bruit_gaussien(self, points):
    points = self.ensure_batch(points)
    if self.random_sigma:
        sigma = np.random.uniform(self.sigma_range_min, self.sigma_range_max, size=(points.shape[0], 1, 1))
    else:
        sigma = self.sigma_range_min
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
       
    Supprime aléatoirement une proportion de points dans un nuage de points,
    avec un taux de dropout tiré d'une loi normale tronquée.

    Args:
        points (np.ndarray): Tableau (N, 3) de points 3D.
        mean_percent (float): Moyenne de la loi normale du taux de dropout.
        std_percent (float): Écart-type de la loi normale.
        max_percent (float): Taux maximum de dropout (en %).
 
    Returns:
        np.ndarray: Nuage de points après dropout.
        
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

def dropout(self, pc:PointCloud):
    if points is None:
        return None
    if points.ndim == 3:
        return np.array([self.dropout_points(p, self.mean_percent_dropout, self.std_percent_dropout, self.max_percent_dropout) for p in points], dtype=object)
    else:
        return self.dropout_points(points, self.mean_percent_dropout, self.std_percent_dropout, self.max_percent_dropout)

def dropout(self, pc:PointCloud):
    p c= pc.downsample()

    self.mean_percent_dropout=mean_percent_dropout
    self.std_percent_dropout=std_percent_dropout
    self.max_percent_dropout=max_percent_dropout
    self.use_dropout=use_dropout
"""