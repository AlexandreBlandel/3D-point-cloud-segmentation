import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pyvista as pv
from abc import ABC, abstractmethod
from metrics.histogram_distance import HistogramDistances

def load_from_ply(source: str):
    """
    Charge un ou plusieurs nuages de points au format .ply.
    
    Paramètre :
        - source (str): chemin vers un fichier .ply ou un dossier contenant des fichiers .ply

    Retourne :
        - points (list[np.ndarray])
        - colors (list[np.ndarray])
        - labels = None
    """
    points_list = []
    colors_list = []
    names=[]

    if os.path.isfile(source) and source.endswith('.ply'):
        # Cas 1 : un seul fichier
        files = [source]
    elif os.path.isdir(source):
        # Cas 2 : dossier contenant des .ply
        files = sorted([
            os.path.join(source, f)
            for f in os.listdir(source)
            if f.endswith('.ply') and os.path.isfile(os.path.join(source, f))
        ])
    else:
        raise ValueError(f"Le chemin {source} n'est ni un fichier .ply ni un dossier contenant des .ply")

    for file in files:
        name=os.path.splitext(os.path.basename(file))[0]
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = np.full_like(points, fill_value=[0.2, 0.5, 1.0])
        points_list.append(points)
        colors_list.append(colors)
        names.append(name)

    labels = None
    return PointClouds(points_list, colors_list, labels, names)

def load_from_npy(source: str,
                  color_0=[0.6, 0.6, 0.6],
                  color_1=[0.0, 0.8, 0.0],
                  color_2=[1.0, 0.0, 0.0]):
    """
    Charge un ou plusieurs fichiers .npy contenant {"points": ..., "labels": ...}
    
    Retourne :
        - points : list[np.ndarray]
        - colors : list[np.ndarray]
        - labels : list[np.ndarray]
    """
    points_list = []
    colors_list = []
    labels_list = []
    names_list = []

    if os.path.isfile(source) and source.endswith('.npy'):
        files = [source]
    elif os.path.isdir(source):
        files = sorted([
            os.path.join(source, f)
            for f in os.listdir(source)
            if f.endswith('.npy') and os.path.isfile(os.path.join(source, f))
        ])
    else:
        raise ValueError(f"Le chemin {source} n'est ni un fichier .npy ni un dossier contenant des .npy")

    for file in files:
        name=os.path.splitext(os.path.basename(file))[0]
        data = np.load(file, allow_pickle=True).item()
        points = data["points"]
        labels = data["labels"]

        colors = np.zeros_like(points)
        colors[labels == 0] = color_0
        colors[labels == 1] = color_1
        colors[labels == 2] = color_2

        points_list.append(np.asarray(points))
        colors_list.append(np.asarray(colors))
        labels_list.append(np.asarray(labels))
        names_list.append(name)

    return PointClouds(points_list, colors_list, labels_list, names_list)

def load_from_dicts(dicts: list, 
                    color_0=[0.6, 0.6, 0.6], 
                    color_1=[0.0, 0.8, 0.0], 
                    color_2=[1.0, 0.0, 0.0]):
    """
    Charge une liste de dictionnaires contenant des nuages de points.

    Chaque dictionnaire doit avoir les clés "points" et "labels".

    Retourne :
        - un objet PointClouds
    """
    points_list = []
    colors_list = []
    labels_list = []
    names_list = []

    for i, d in enumerate(dicts):
        points = np.asarray(d["points"])
        labels = np.asarray(d["labels"])

        colors = np.zeros_like(points)
        colors[labels == 0] = color_0
        colors[labels == 1] = color_1
        colors[labels == 2] = color_2

        points_list.append(points)
        colors_list.append(colors)
        labels_list.append(labels)
        names_list.append(f"dictpcd_{i:04d}")

    return PointClouds(points_list, colors_list, labels_list, names_list)

class PointClouds(ABC):
    def __init__(self, points: list, colors: list, labels: list = None, names=None):
        """
        - points: list of (N_i, 3) np.ndarrays
        - colors: list of (N_i, 3) np.ndarrays
        - labels: list of (N_i,) np.ndarrays or None
        - names: list of str or a single str or None
        """
        assert len(points) == len(colors)
        self.points = points
        self.colors = colors
        self.labels = labels if labels is not None else [None] * len(points)
        self.B = len(points)

        if isinstance(names, str):
            self.names = [names] * self.B
        elif names is None:
            self.names = [f"pointcloud_{i}" for i in range(self.B)]
        else:
            assert len(names) == self.B
            self.names = names

    def __getitem__(self, idx: int):
        return self.__class__(
            [self.points[idx]],
            [self.colors[idx]],
            [self.labels[idx]] if self.labels[idx] is not None else None,
            [self.names[idx]]
        )
    
    def __len__(self):
        return len(self.points)

    def save_labelled_point_clouds(self, save_repository):
        os.makedirs(save_repository, exist_ok=True)
        for i in range(self.B):
            name=self.names[i]
            if self.labels[i] is None:
                filename = os.path.join(save_repository, f"{name}.ply")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.asarray(self.points[i]))
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(self.colors[i]))
                o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
                print(f"✅ Nuage sauvegardé en .ply : {filename}")
            else:
                filename = os.path.join(save_repository, f"{name}.npy")
                np.save(f"{filename}",{"points":self.points[i], "labels":self.labels[i]})
                print(f"✅ Nuage sauvegardé en .npy : {filename}")

    def normalize(self):
        new_points = []
        for pts in self.points:
            centroid = np.mean(pts, axis=0)
            centered = pts - centroid
            max_dist = np.max(np.linalg.norm(centered, axis=1))
            normalized = (centered / max_dist + 1) / 2
            new_points.append(normalized)
        self.points = new_points
        return self
    
    def compute_histograms(self, 
                          indicator, 
                          bins:int =50, 
                          range=None,
                          normalize_points=False, 
                          normalize_hist=True,):
        """ 
        Computes all the histograms of the batch for the indicator
        """
        hists=[]
        bin_edges=[]
        for i in range(self.B):
            pc_i=self.__getitem__(i)
            if normalize_points:
                pc_i=pc_i.normalize()
            values=indicator.compute(pc_i)
            hist, bin_edge = np.histogram(values, bins=bins, range=range)
            if normalize_hist:
                hist = HistogramDistances.normalize_hist(hist)
            hists.append(hist)
            bin_edges.append(bin_edge)
        
        return hists, bin_edges
    
    @staticmethod
    def plot_single_histogram(hist, bin_edges, name:str, indicator):
        """
        static_method (so that computation and plot of the histogram are separated. plot doesn't need to compute the histogram of self). 
        """
        
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.bar(
            bin_centers,
            hist,
            width=bin_edges[1] - bin_edges[0],
            alpha=0.7,
            align="center",
        )
        plt.title(f"Histogramme de {indicator.__class__.__name__} du nuage de points ({name})")
        plt.xlabel("Valeur")
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.savefig(f"figures/{indicator.__class__.__name__}_{name}.png")
        plt.close()

    def visualiser(self, idx=0, filename=None):
        """
        Visualise le nuage de points à l'index `idx` du batch.
        Enregistre un fichier HTML dans le dossier `figures`.
        """
        if idx >= self.B:
            raise IndexError(f"Index {idx} hors limites pour un batch de taille {self.B}")

        points = np.asarray(self.points[idx])
        colors = np.clip(np.asarray(self.colors[idx]), 0, 1)

        cloud = pv.PolyData(points)
        cloud["colors"] = (colors * 255).astype(np.uint8)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_points(cloud, scalars="colors", rgb=True, point_size=5)
        plotter.background_color = "white"
        plotter.add_title(f"Nuage de points - {self.names[idx]}", font_size=12)

        os.makedirs("figures", exist_ok=True)
        if filename is None:
            filename = self.names[idx]
        html_path = f"figures/{filename}.html"
        plotter.export_html(html_path)
        print(f" Visualisation HTML enregistrée : {html_path}")
        plotter.clear()
        plotter.close()


    def duplicate(self, n: int, suffix='_dup'):
        """
        Duplique chaque élément du batch n fois.
        Optimisé sans boucle imbriquée.
        """
        new_points = sum([[p.copy()] * n for p in self.points], [])
        new_colors = sum([[c.copy()] * n for c in self.colors], [])
        new_labels = sum(
            [[l.copy()] * n if l is not None else [None] * n for l in self.labels],
            []
        )
        new_names = sum([[f"{name}{suffix}_{i:04d}"] * n for i, name in enumerate(self.names)], [])

        return self.__class__(new_points, new_colors, new_labels, new_names)

    def fusion(self, other, names=None):
        """
        Fusionne chaque nuage de self[i] avec other[i] (doivent avoir même taille B).
        Concatène points, couleurs, et labels (si cohérents).
        """
        if len(self) != len(other):
            raise ValueError("Les deux batches doivent avoir la même taille pour une fusion par échantillon.")

        fused_points = []
        fused_colors = []
        fused_labels = []
        fused_names = []

        for i in range(self.B):
            pts1, pts2 = self.points[i], other.points[i]
            cols1, cols2 = self.colors[i], other.colors[i]
            labs1, labs2 = self.labels[i], other.labels[i]

            pts = np.vstack([pts1, pts2])
            cols = np.vstack([cols1, cols2])

            if labs1 is not None and labs2 is not None:
                labs = np.concatenate([labs1, labs2])
            elif labs1 is None and labs2 is None:
                labs = None
            else:
                raise TypeError("Impossible de fusionner un nuage avec labels et un autre sans labels.")

            name = f"{self.names[i]}_fused_with_{other.names[i]}" if names is None else names[i]

            fused_points.append(pts)
            fused_colors.append(cols)
            fused_labels.append(labs)
            fused_names.append(name)

        return self.__class__(fused_points, fused_colors, fused_labels, fused_names)

    def sizes(self):
        """
        Retourne une liste contenant le nombre de points dans chaque nuage du batch.
        """
        return [p.shape[0] for p in self.points]
    
    def split(self, ratio:int):
        split_idx = int(self.B * ratio)
        return self.__class__(
            self.points[:split_idx],
            self.colors[:split_idx],
            self.labels[:split_idx],
            self.names[:split_idx]
        ), self.__class__(
            self.points[split_idx:],
            self.colors[split_idx:],
            self.labels[split_idx:],
            self.names[split_idx:]
        )
    
    def concat(self, other):
        """
        Concatène deux objets PointClouds (de tailles B1 et B2) en un seul batch de taille B1 + B2.

        Retourne :
            - un nouvel objet PointClouds
        """
        # Sanity check
        if not isinstance(other, PointClouds):
            raise TypeError("On ne peut concaténer qu'avec un autre PointClouds.")

        # Concaténation des champs
        new_points = self.points + other.points
        new_colors = self.colors + other.colors
        new_labels = self.labels + other.labels
        new_names = self.names + other.names

        return self.__class__(new_points, new_colors, new_labels, new_names)