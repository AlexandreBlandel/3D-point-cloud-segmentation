import numpy as np


class HistogramDistance:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

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

    def l1(self, h1, h2):
        """Distance de Manhattan (L1)"""
        return np.sum(np.abs(h1 - h2))

    def l2(self, h1, h2):
        """Distance Euclidienne (L2)"""
        return np.sqrt(np.sum((h1 - h2) ** 2))

    def chi_square(self, h1, h2):
        """Distance du Chi²"""
        denom = h1 + h2 + self.epsilon
        num = (h1 - h2) ** 2
        return np.sum(num / denom)

    def kl_divergence(self, h1, h2):
        """Divergence de Kullback-Leibler (asymétrique)"""
        h1 = h1 + self.epsilon
        h2 = h2 + self.epsilon
        return np.sum(h1 * np.log(h1 / h2))

    def js_divergence(self, h1, h2):
        """Divergence de Jensen-Shannon (symétrique)"""
        h1 = h1 + self.epsilon
        h2 = h2 + self.epsilon
        m = 0.5 * (h1 + h2)
        return 0.5 * self.kl_divergence(h1, m) + 0.5 * self.kl_divergence(h2, m)
