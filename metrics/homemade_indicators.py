"""
Homemade indicators
Ce fichier permet de code toutes les fonctions qui vont être utilisées pour analyser les nuages de point.
"""

import numpy as np
from scipy.spatial.distance import pdist


def pair_distances(point_cloud):
    # pdist calcule toutes les distances par paire, retourne un vecteur de taille N*(N-1)/2
    return pdist(point_cloud, metric="euclidean")


def mean_distance_between_points(pair_distances):
    """
    Calcule la distance moyenne entre tous les points d'un nuage de points.

    Paramètres :
        point_cloud (np.ndarray): Nuage de points de forme (N, 3)

    Retour :
        float : Distance moyenne entre les points
    """
    return pair_distances.mean()


from scipy.spatial import KDTree


def nearest_neighbor_distance(point_cloud):
    if point_cloud.shape[0] < 2:
        return 0.0

    tree = KDTree(point_cloud)
    # k=2 renvoie le point lui-même (distance 0) et son plus proche voisin
    distances, _ = tree.query(point_cloud, k=2)
    # On prend la distance au plus proche voisin (pas soi-même)
    nearest_distances = distances[:, 1]
    return nearest_distances


def average_nearest_neighbor_distance(nearest_distances):
    return nearest_distances.mean()


def histogramme_min_distance(nearest_distances, bins=10, range=None, density=False):
    """
    Calcule l'histogramme des distances au plus proche voisin. Permet de détecter si il y a trop d'outliers.

    Paramètres :
        nearest_distances (np.ndarray) : Tableau des distances au plus proche voisin
        bins (int ou sequence) : Nombre de bins ou bornes des bins
        range (tuple) : Étendue des valeurs (min, max) pour les bins
        density (bool) : Si True, normalise l'histogramme pour que l'aire soit 1

    Retour :
        hist (np.ndarray) : Comptes ou densités dans chaque bin
        bin_edges (np.ndarray) : Bords des bins
    """
    hist, bin_edges = np.histogram(
        nearest_distances, bins=bins, range=range, density=density
    )
    return hist, bin_edges


def histogramme_axis_y(point_cloud, bins=10, range=None, density=False):
    """
    Calcule l'histogramme des coordonnées y d'un nuage de points.

    Paramètres :
        point_cloud (np.ndarray) : Tableau Nx3 des coordonnées
        bins (int ou sequence) : Nombre ou définition des intervalles (par défaut 10)
        range (tuple) : Étendue (min, max) des valeurs sur y (optionnel)
        density (bool) : Si True, normalise l'histogramme (aire = 1)

    Retour :
        hist (np.ndarray) : Valeurs de l'histogramme
        bin_edges (np.ndarray) : Bords des bins
    """
    y_coords = point_cloud[:, 1]  # On extrait l'axe Y
    hist, bin_edges = np.histogram(y_coords, bins=bins, range=range, density=density)
    return hist, bin_edges


def histogramme_distance_axe_y(point_cloud, bins=10, range=None, density=False):
    """
    Calcule l'histogramme des distances des points à l'axe Y (vertical).

    Paramètres :
        point_cloud (np.ndarray) : Tableau Nx3 des coordonnées
        bins (int ou array) : Nombre ou définition des intervalles (par défaut 10)
        range (tuple) : Étendue (min, max) des distances (optionnel)
        density (bool) : Si True, normalise l'histogramme (aire = 1)

    Retour :
        hist (np.ndarray) : Valeurs de l'histogramme
        bin_edges (np.ndarray) : Bords des bins
    """
    distances = np.sqrt(point_cloud[:, 0] ** 2 + point_cloud[:, 2] ** 2)
    hist, bin_edges = np.histogram(distances, bins=bins, range=range, density=density)
    return hist, bin_edges


"""brouillon:"""


class Histogramme:
    def __init__(self, bins=32, normalize=True):
        self.bins = bins
        self.normalize = normalize

    def compute_histogram(self, pc):
        """
        Convertit un nuage de points en un histogramme 3D concaténé.
        """
        if self.normalize:
            pc = pc - np.mean(pc, axis=0)
            pc = pc / np.max(np.linalg.norm(pc, axis=1))

        # Histogramme pour chaque axe
        hist_x, _ = np.histogram(pc[:, 0], bins=self.bins, range=(-1, 1), density=True)
        hist_y, _ = np.histogram(pc[:, 1], bins=self.bins, range=(-1, 1), density=True)
        hist_z, _ = np.histogram(pc[:, 2], bins=self.bins, range=(-1, 1), density=True)
        return np.concatenate([hist_x, hist_y, hist_z])


from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import entropy


class DistanceHistogramme(Histogramme):
    def l2(self, pc1, pc2):
        h1 = self.compute_histogram(pc1)
        h2 = self.compute_histogram(pc2)
        return euclidean(h1, h2)

    def l1(self, pc1, pc2):
        h1 = self.compute_histogram(pc1)
        h2 = self.compute_histogram(pc2)
        return cityblock(h1, h2)

    def kl(self, pc1, pc2):
        h1 = self.compute_histogram(pc1)
        h2 = self.compute_histogram(pc2)
        # Ajouter un epsilon pour éviter les 0
        eps = 1e-10
        return entropy(h1 + eps, h2 + eps)
