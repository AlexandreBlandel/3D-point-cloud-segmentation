import numpy as np
import ot
from abc import ABC, abstractmethod
    
class HistogramDistances(ABC):
    @abstractmethod
    def compute(self, hists1, hists2):
        pass

    @staticmethod
    def normalize_hist(hist):
        hist = hist.astype(np.float64)
        total = np.sum(hist)
        if total > 0:
            return hist / total
        return hist
    
    def pairwise_distance(self, hists1, hists2, dist_fn):
        """
        Calcule la matrice (B1 x B2) des distances entre les histogrammes hists1 et hists2,
        en appliquant dist_fn(h1, h2) à chaque couple (i, j).
        """
        B1 = len(hists1)
        B2 = len(hists2)
        distances = [[None for _ in range(B2)] for _ in range(B1)]

        for i in range(B1):
            h1 = hists1[i]
            for j in range(B2):
                h2 = hists2[j]
                distances[i][j] = dist_fn(h1, h2)

        return distances
    
    def mean_distance(self, distances):
        return np.mean(distances)


    
class l1_hist_dist(HistogramDistances):
    def compute(self, hists1, hists2):
        return self.pairwise_distance(hists1, hists2, lambda h1, h2: np.sum(np.abs(h1 - h2)))

    
class l2_hist_dist(HistogramDistances):
    def compute(self, hists1, hists2):
        return self.pairwise_distance(hists1, hists2, lambda h1, h2: np.sqrt(np.sum((h1 - h2)**2)))

    
class chi2_hist_dist(HistogramDistances):
    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon

    def compute(self, hists1, hists2):
        def chi2(h1, h2):
            denom = h1 + h2 + self.epsilon
            num = (h1 - h2) ** 2
            return np.sum(num / denom)
        return self.pairwise_distance(hists1, hists2, chi2)

    
class kl_divergence(HistogramDistances):
    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon

    def compute(self, hists1, hists2):
        def kl(h1, h2):
            h1 = h1 + self.epsilon
            h2 = h2 + self.epsilon
            return np.sum(h1 * np.log(h1 / h2))
        return self.pairwise_distance(hists1, hists2, kl)

    
class js_divergence(HistogramDistances):
    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon

    def compute(self, hists1, hists2):
        def kl(h1, h2):
            h1 = h1 + self.epsilon
            h2 = h2 + self.epsilon
            return np.sum(h1 * np.log(h1 / h2))
        
        def js(h1, h2):
            h1 = h1 + self.epsilon
            h2 = h2 + self.epsilon
            m = 0.5 * (h1 + h2)
            return 0.5 * kl(h1, m) + 0.5 * kl(h2, m)
        
        return self.pairwise_distance(hists1, hists2, js)

    
class Earth_Mover_hist_dist(HistogramDistances):
    def __init__(self, bin_positions=None):
        self.bin_positions = bin_positions

    def compute(self, hists1, hists2):
        def emd(h1, h2):
            n = len(h1)
            if self.bin_positions is None:
                pos = np.arange(n).reshape(-1, 1)
            else:
                pos = self.bin_positions
            M = ot.dist(pos, pos, metric="euclidean")
            emd_value = ot.emd2(h1, h2, M)
            return np.sqrt(emd_value)
        
        return self.pairwise_distance(hists1, hists2, emd)

    
class Earth_Mover_hist_dist_v2(HistogramDistances):
    def compute(self, hists1, hists2):
        def emd_cdf(h1, h2):
            h1 = self.normalize_hist(h1)
            h2 = self.normalize_hist(h2)
            cdf1 = np.cumsum(h1)
            cdf2 = np.cumsum(h2)
            return np.sum(np.abs(cdf1 - cdf2))

        return self.pairwise_distance(hists1, hists2, emd_cdf)


    
"""
class HistogramDistance:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    @staticmethod
    def normalize_hist(hist):
        
        Normalise un histogramme pour que la somme des fréquences soit 1.
        
        hist = hist.astype(np.float64)
        total = np.sum(hist)
        if total > 0:
            return hist / total
        return hist

    def l1(self, h1, h2):
        return np.sum(np.abs(h1 - h2))

    def l2(self, h1, h2):
        Distance Euclidienne (L2)
        return np.sqrt(np.sum((h1 - h2) ** 2))

    def chi_square(self, h1, h2):
        Distance du Chi²
        denom = h1 + h2 + self.epsilon
        num = (h1 - h2) ** 2
        return np.sum(num / denom)

    def kl_divergence(self, h1, h2):
        Divergence de Kullback-Leibler (asymétrique)
        h1 = h1 + self.epsilon
        h2 = h2 + self.epsilon
        return np.sum(h1 * np.log(h1 / h2))

    def js_divergence(self, h1, h2):
        Divergence de Jensen-Shannon (symétrique)
        h1 = h1 + self.epsilon
        h2 = h2 + self.epsilon
        m = 0.5 * (h1 + h2)
        return 0.5 * self.kl_divergence(h1, m) + 0.5 * self.kl_divergence(h2, m)

    def emd(self, h1, h2, bin_positions=None):
        h1 = h1 / np.sum(h1)
        h2 = h2 / np.sum(h2)

        n = len(h1)
        if bin_positions is None:
            bin_positions = np.arange(n).reshape(-1, 1)

        M = ot.dist(bin_positions, bin_positions, metric="euclidean")
        emd_value = ot.emd2(h1, h2, M)
        return np.sqrt(emd_value)
    
    def emd_v2(self, h1, h2):
        Earth Mover Distance (EMD) pour histogrammes 1D
        h1 = self.normalize_hist(h1)
        h2 = self.normalize_hist(h2)
        cdf1 = np.cumsum(h1)
        cdf2 = np.cumsum(h2)
        return np.sum(np.abs(cdf1 - cdf2))
"""