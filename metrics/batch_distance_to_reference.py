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


class Distance_to_ref(ABC):
    def __init__(self, synthe_repository:str, reel_repository: str, distance:Distance, max_size_repository=500):
        self.synthe_repository=synthe_repository
        self.reel_repository=reel_repository
        self.distance=distance
        self.distance_values=None
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
        return synthe_files, reel_files

    def generate_batches(self, files, batch_size):
        """Génère des batches de PointClouds et leur indice de départ"""
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            pcs = [npypcd(f) if f.endswith(".npy") else plypcd(f) for f in batch_files]
            yield i, pcs  # pcs = liste de PointCloud, i = indice global de départ


    def compute(self):
        synthe_pcs, reel_pcs=self.load()
        self.distance_values = np.zeros((len(reel_pcs), len(synthe_pcs)))
        for i, reel_batch in self.generate_batches(reel_pcs, self.max_size_repository):
            for j, synthe_batch in self.generate_batches(synthe_pcs, self.max_size_repository):
                for i_local, pc_reel in enumerate(reel_batch):
                    i_global=i + i_local
                    for j_local, pc_synthe in enumerate(synthe_batch):
                        j_global=j+j_local
                        self.distance_values[i_local, j_local] = self.distance.compute(pc_reel, pc_synthe)

        return self.distance_values

    def mean(self):
        """return la valeur moyenne du tableau des métriques de similarite"""
        if self.distance_values is None:
            raise ValueError("You must call compute() before mean()")
        return np.mean(self.distance_values)
    
