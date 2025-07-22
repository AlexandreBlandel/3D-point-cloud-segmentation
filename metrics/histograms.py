import matplotlib.pyplot as plt
import numpy as np
from metrics.indicators import Indicators


class PointCloudHistograms:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
        self.indicators = Indicators(point_cloud)

    def compute_histogram(
        self, indicator_name: str, bins: int = 50, range=None, normalize=True
    ):
        """
        Calcule l'histogramme de l'indicateur donné.
        """
        if indicator_name not in Indicators.list_available():
            raise ValueError(
                f"Indicateur inconnu : {indicator_name}. "
                f"Choisissez parmi : {Indicators.list_available()}"
            )
        values = self.indicators.compute(indicator_name)
        hist, bin_edges = np.histogram(values, bins=bins, range=range)
        if normalize:
            hist = self.normalize_hist(hist)
        return hist, bin_edges

    def plot_histogram(
        self, indicator_name: str, bins: int = 50, range=None, hist_data=None
    ):
        """
        Affiche l'histogramme de l'indicateur donné.
        """
        if hist_data is None:
            hist, bin_edges = self.compute_histogram(
                indicator_name, bins=bins, range=range
            )
        else:
            hist, bin_edges = hist_data
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        plt.bar(
            bin_centers,
            hist,
            width=bin_edges[1] - bin_edges[0],
            alpha=0.7,
            align="center",
        )
        plt.title(f"Histogramme de {indicator_name}")
        plt.xlabel("Valeur")
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.show()

    @staticmethod
    def normalize_hist(hist):
        """
        Normalise un histogramme pour que la somme des fréquences soit 1.
        """
        hist = hist.astype(np.float64)
        total = np.sum(hist)
        if total > 0:
            return hist / total
        return hist
