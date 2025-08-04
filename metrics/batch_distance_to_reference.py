"""
fichier pour implémenter une classe calculant les distances etc entre les nuages de point synthétique et les nuages de points réels
"""
import numpy as np
import os
from point_cloud import PointCloud, npypcd, plypcd, dictpcd
from metrics.metrics import Distance
from metrics.indicators import Indicator
from metrics.histogram_distance import HistogramDistance
from abc import ABC, abstractmethod


class Metrics_to_ref(ABC):
    def __init__(self, synthe_repository:str, reel_repository: str, metrics, max_size_repository=500):
        self.synthe_repository=synthe_repository
        self.reel_repository=reel_repository
        self.metrics=metrics
        self.metrics_values=None
        self.max_size_repository=500

    def load(self):
        synthe_files = sorted([
            os.path.join(self.synthe_repository, f) 
            for f in os.listdir(self.synthe_repository) 
            if f.endswith(".npy") or f.endswith(".ply")
        ])
        reel_files = sorted([
            os.path.join(self.reel_repository, f) 
            for f in os.listdir(self.reel_repository) 
            if f.endswith(".npy") or f.endswith(".ply")
        ])
        synthe_pcs=[npypcd(source) for source in synthe_files]
        reel_pcs=[plypcd(source) for source in reel_files]
        return synthe_pcs, reel_pcs

    @abstractmethod
    def compute(self):
        """return le tableau de la valeur des métriques de similarite entre pc synthetique (colonne) et pc reel (ligne)"""
        pass

    def mean(self):
        """return la valeur moyenne du tableau des métriques de similarite"""
        if self.metrics_values is None:
            raise ValueError("You must call compute() before mean()")
        return np.mean(self.metrics_values)
    

class Distance_to_ref(Metrics_to_ref):
    def __init__(self, synthe_repository:str, reel_repository: str, distance: Distance):
        super().__init__(synthe_repository=synthe_repository, reel_repository=reel_repository, metrics=distance, max_size_repository=500)

    def compute(self):
        synthe_pcs, reel_pcs=self.load()
        self.metrics_values = np.zeros((len(reel_pcs), len(synthe_pcs)))
        for i, pc_reel in enumerate(reel_pcs):
            for j, pc_synthe in enumerate(synthe_pcs):
                self.metrics_values[i, j] = self.metrics.compute(pc_reel, pc_synthe)

        return self.metrics_values

    
class Indicator_to_ref(ABC):
    def __init__(self, synthe_repository:str, reel_repository: str, indicator: Indicator, hist_dist: HistogramDistance):
        super().__init__(synthe_repository=synthe_repository, reel_repository=reel_repository, metrics=indicator)
        self.hist_dist=hist_dist

    def compute(self):
        synthe_pcs, reel_pcs=self.load()
        self.metrics_values = np.zeros((len(reel_pcs), len(synthe_pcs)))
        for i, pc_reel in enumerate(reel_pcs):
            for j, pc_synthe in enumerate(synthe_pcs):
                hist_i = self.metrics.compute(pc_synthe)
                hist_j = self.metrics.compute(pc_reel)
                self.metrics_values[i, j]=self.hist_dist.compute(hist_i, hist_j)

        return self.metrics_values
    

    
