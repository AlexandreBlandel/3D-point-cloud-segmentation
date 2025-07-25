import numpy as np
import pyvista as pv
import trimesh
import pyransame
import open3d as o3d
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from metrics.indicators import Indicator
from metrics.histogram_distance import HistogramDistance

class PointCloud(ABC):
    def __init__(self, source):
        self.points, self.colors = self.load(source)

    @abstractmethod
    def load(self, source):
        pass

    @staticmethod
    def infer_labels_from_colors(colors, tol=1e-2):
        labels = np.full(colors.shape[0], -1, dtype=int)
        mask_gray = np.all(np.abs(colors - [0.6, 0.6, 0.6]) < tol, axis=1)
        mask_green = np.all(np.abs(colors - [0.0, 0.8, 0.0]) < tol, axis=1)
        mask_red = np.all(np.abs(colors - [1.0, 0.0, 0.0]) < tol, axis=1)
        labels[mask_gray] = 0
        labels[mask_green] = 1
        labels[mask_red] = 2
        return labels

    @abstractmethod
    def get_labels(self):
        pass

    @property
    def size(self):
        if len(self.points) != len(self.colors):
            raise ValueError("les points et les couleurs ne sont pas de même taille")
        return len(self.points)

    def visualiser(self, return_pc=False):
        # Créer un nuage de points Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        o3d.visualization.draw_geometries([pcd])
        if return_pc:
            return pcd

    @staticmethod
    def normalize_points(points: np.ndarray) -> np.ndarray:

        # Étape 1 : Centrage autour du barycentre
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid

        # Étape 2 : Mise à l’échelle (normalisation par la plus grande norme)
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        points_normalized = points_centered / max_dist

        # Étape 3 : Mise à l’échelle dans le cube [0, 1]^3
        return (points_normalized + 1) / 2
    
    def normalize(self):
        self.points = self.normalize_points(self.points)

    def downsample(self, ratio=0.05, max_points=25_000):
        points = np.asarray(self.points)
        colors = np.asarray(self.colors)
        n = len(points)
        m = int(min(ratio * n, max_points))
        idx = np.random.choice(n, m, replace=False)
        self.points = points[idx]
        self.colors = colors[idx]

    @staticmethod
    def rotate_points(
        points: np.ndarray, axis="x", degrees=-90, center=(0, 0, 0)
    ) -> np.ndarray:
        """
        Applique une rotation à un nuage de points représenté par un tableau NumPy.

        Paramètres :
        - points : np.ndarray de forme (N, 3)
        - axis : "x", "y" ou "z"
        - degrees : angle en degrés
        - center : tuple (x, y, z), centre de rotation

        Retourne :
        - points : np.ndarray de forme (N, 3) après rotation
        """
        radians = np.deg2rad(degrees)
        center = np.asarray(center)

        if axis == "x":
            R = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(radians), -np.sin(radians)],
                    [0, np.sin(radians), np.cos(radians)],
                ]
            )
        elif axis == "y":
            R = np.array(
                [
                    [np.cos(radians), 0, np.sin(radians)],
                    [0, 1, 0],
                    [-np.sin(radians), 0, np.cos(radians)],
                ]
            )
        elif axis == "z":
            R = np.array(
                [
                    [np.cos(radians), -np.sin(radians), 0],
                    [np.sin(radians), np.cos(radians), 0],
                    [0, 0, 1],
                ]
            )
        else:
            raise ValueError("Axe invalide. Choisir parmi 'x', 'y' ou 'z'.")

        return (points - center) @ R.T + center

    def rotate(self, axis="x", degrees=-90, center=(0, 0, 0)):
        self.points = self.rotate_points(
            self.points, axis=axis, degrees=degrees, center=center
        )

    @staticmethod
    def compute_histogram_points(points: np.ndarray, indicator_cls: type[Indicator], bins: int = 50, range=None, normalize_points=False, normalize_histo=True):
        indicator = indicator_cls(points, normalize=normalize_points)
        values = indicator.compute()
        hist, bin_edges = np.histogram(values, bins=bins, range=range)
        if normalize_histo:
            hist = HistogramDistance.normalize_hist(hist)
        return hist, bin_edges

    def compute_histogram(
        self,
        indicator_cls: type[Indicator],
        bins: int = 50,
        range=None,
        normalize_points=False,
        normalize_histo=True,
    ):
        return PointCloud.compute_histogram_points(
            self.points,
            indicator_cls,
            bins=bins,
            range=range,
            normalize_points=normalize_points,
            normalize_histo=normalize_histo,
        )
    
    @staticmethod
    def plot_histogram(hist, bin_edges, indicator_cls: type[Indicator]):
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.bar(
            bin_centers,
            hist,
            width=bin_edges[1] - bin_edges[0],
            alpha=0.7,
            align="center",
        )
        plt.title(f"Histogramme de {indicator_cls.__name__}")
        plt.xlabel("Valeur")
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.show()



"""  Pour la classe generators:      
    def ajout_bruit(self):

    def apply_translation(self):

    def apply_rotation(self):

    def apply_random_rotation(self):

    def apply_scaling(self):
    
    def deformation_tot(self):

    def dropout_point_cloud(self):

    def generate_sphere_points(self):
"""


class plypcd(PointCloud):
    def __init__(self, source):
        super().__init__(source)

    def load(self, source):
        pcd = o3d.io.read_point_cloud(source)
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = np.full_like(points, fill_value=[0.2, 0.5, 1.0])
        return points, colors

    def get_labels(self):
        raise ValueError(
            f"Impossible d'extraire des labels pour les nuages de points réels"
        )


class npypcd(PointCloud):
    def __init__(self, source):
        super().__init__(source)

    def load(self, source):
        data = np.load(source, allow_pickle=True).item()
        points = data["points"]
        labels = data["labels"]

        colors = np.zeros_like(points)
        colors[labels == 0] = [0.6, 0.6, 0.6]  # gris pour corolle
        colors[labels == 1] = [0.0, 0.8, 0.0]  # vert pour plante
        colors[labels == 2] = [1.0, 0.0, 0.0]  # rouge pour outliers

        return np.asarray(points), np.asarray(colors)

    def get_labels(self):
        return self.infer_labels_from_colors(self.colors)


class dictpcd(PointCloud):
    def __init__(self, source):
        super().__init__(source)

    def load(self, source):
        points = source["points"]
        labels = source["labels"]

        colors = np.zeros_like(points)
        colors[labels == 0] = [0.6, 0.6, 0.6]  # gris pour corolle
        colors[labels == 1] = [0.0, 0.8, 0.0]  # vert pour plante
        colors[labels == 2] = [1.0, 0.0, 0.0]  # rouge pour outliers

        return np.asarray(points), np.asarray(colors)

    def get_labels(self):
        return self.infer_labels_from_colors(self.colors)


class o3dpcd(PointCloud):
    def __init__(self, source):
        super().__init__(source)

    def load(self, source):
        points = np.asarray(source.points)
        colors = np.asarray(source.colors)
        return points, colors

    def get_labels(self):
        return self.infer_labels_from_colors(self.colors)
