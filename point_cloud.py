import numpy as np
import pyvista as pv
import trimesh
import pyransame
import open3d as o3d
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from metrics.histogram_distance import HistogramDistances
from mpl_toolkits.mplot3d import Axes3D  

class PointCloud(ABC):
    def __init__(self, source):
        self.name=self.extract_name(source)
        self.points, self.colors = self.load(source)
        self.labels=self.get_labels()

    @abstractmethod
    def load(self, source):
        pass

    @abstractmethod
    def save_labeled_point_cloud(self, save_repository):
        """
        Enregistre le PointCloud au format .npy (.ply) si c'est un sans labels
        """
        pass

    @staticmethod
    def get_labels_from_colors(colors, color_0, color_1, color_2, tol=1e-2):
        labels = np.full(colors.shape[0], -1, dtype=int)
        mask_gray = np.all(np.abs(colors - color_0) < tol, axis=1)
        mask_green = np.all(np.abs(colors - color_1) < tol, axis=1)
        mask_red = np.all(np.abs(colors - color_2) < tol, axis=1)
        labels[mask_gray] = 0
        labels[mask_green] = 1
        labels[mask_red] = 2
        return labels

    def get_labels(self):
        return None

    @property
    def size(self):
        if self.points.shape != self.colors.shape:
            raise ValueError("les points et les couleurs ne sont pas de même taille")
        return len(self.points)

    @staticmethod
    def extract_name(source):
        if isinstance(source, str):
            return os.path.splitext(os.path.basename(source))[0]
        return "pointcloud"


    def visualiser(self, save=True, filename=None):
        if filename is None:
            filename=self.name
        points = np.asarray(self.points)
        colors = np.clip(np.asarray(self.colors), 0, 1)

        # Créer l'objet PolyData
        cloud = pv.PolyData(points)
        cloud["colors"] = (colors * 255).astype(np.uint8)

        # Préparer le plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.add_points(cloud, scalars="colors", rgb=True, point_size=5)
        plotter.background_color = "white"
        plotter.add_title(f"Nuage de points - {self.name}", font_size=12)

        if save:
            os.makedirs("figures", exist_ok=True)
            html_path = f"figures/{filename}.html"
            plotter.export_html(html_path)
            print(f" Visualisation HTML enregistrée dans : {html_path}")

    def normalize(self):
        """
        normalize in [0, 1]^3
        """

        def centering(points):
            centroid = np.mean(points, axis=0)
            points_centered = points - centroid
            return points_centered
        self.points=centering(self.points)

        def scaling_in_unit_cube(points):
            max_dist = np.max(np.linalg.norm(points, axis=1))
            points_normalized = points / max_dist
            points = (points_normalized + 1) / 2
            return points
        self.points=scaling_in_unit_cube(self.points)

        return self
        

    def compute_histogram(
        self,
        indicator,
        bins: int = 50,
        range=None,
        normalize_points=False,
        normalize_hist=True,
    ):
        if normalize_points:
            self=self.normalize()
        values=indicator.compute(self)
        hist, bin_edges = np.histogram(values, bins=bins, range=range)
        if normalize_hist:
            hist = HistogramDistances.normalize_hist(hist)
        return hist, bin_edges


    @staticmethod
    def plot_histogram(hist, bin_edges, name:str, indicator):
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
        
    def duplicate(self, n: int):
        from batch_point_cloud import BatchPointCloud
        """
        Duplique les points et les couleurs `n` fois pour obtenir un batch vectorisé de shape (n, N, 3).
        """
        points = np.tile(self.points[None, :, :], (n, 1, 1))  # (n, N, 3)
        colors = np.tile(self.colors[None, :, :], (n, 1, 1))  # (n, N, 3)
        if self.labels is not None:
            labels=np.tile(self.labels[None, :], (n, 1))
        else:
            labels=None
        batch=BatchPointCloud(points, colors, labels)
        return batch
    
    def fusion(self, pc, name = None):
        points = np.vstack([self.points, pc.points])
        colors = np.vstack([self.colors, pc.colors])
        if self.labels is not None:
            if pc.labels is None:
                raise TypeError("can't fuse reel point cloud (no labels) with synthetic point cloud")
            else:
                labels=np.concatenate([self.labels, pc.labels])
        else:
            if pc.labels is not None:
                raise TypeError("can't fuse reel point cloud (no labels) with synthetic point cloud")
            else:
                labels=None
        if name is not None:
            self.name=name
        self.points=points
        self.colors=colors
        self.labels=labels
        return self


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

    def save_labeled_point_cloud(self, save_repository):
        os.makedirs(save_repository, exist_ok=True)
        filename = os.path.join(save_repository, f"{self.name}.ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.points))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(self.colors))
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
        print(f"✅ Nuage sauvegardé en .ply : {filename}")



class npypcd(PointCloud):
    def __init__(self, source, color_0=[0.6, 0.6, 0.6], color_1=[0.0, 0.8, 0.0], color_2=[1.0, 0.0, 0.0] ):
        self.color_0=color_0
        self.color_1=color_1
        self.color_2=color_2        
        super().__init__(source)

    def load(self, source):
        data = np.load(source, allow_pickle=True).item()
        points = data["points"]
        labels = data["labels"]

        colors = np.zeros_like(points)
        colors[labels == 0] = self.color_0  
        colors[labels == 1] = self.color_1  
        colors[labels == 2] = self.color_2  

        return np.asarray(points), np.asarray(colors)

    def get_labels(self):
        return self.get_labels_from_colors(self.colors, self.color_0, self.color_1, self.color_2)

    def save_labeled_point_cloud(self, save_repository):
        os.makedirs(save_repository, exist_ok=True)
        filename=os.path.join(save_repository, f"{self.name}")
        labels=self.get_labels()
        np.save(f"{filename}.npy",{"points":self.points, "labels":labels})
        print(f"✅ Nuage sauvegardé en .npy : {filename}")

class dictpcd(PointCloud):
    def __init__(self, source, name="pointcloud", color_0=[0.6, 0.6, 0.6], color_1=[0.0, 0.8, 0.0], color_2=[1.0, 0.0, 0.0]):
        self.color_0=color_0
        self.color_1=color_1
        self.color_2=color_2
        super().__init__(source)
        self.name=name

    def load(self, source):
        points = source["points"]
        labels = source["labels"]

        colors = np.zeros_like(points)
        colors[labels == 0] = self.color_0  
        colors[labels == 1] = self.color_1  
        colors[labels == 2] = self.color_2  

        return np.asarray(points), np.asarray(colors)

    def get_labels(self, tol=1e-2):
        return self.get_labels_from_colors(self.colors, self.color_0, self.color_1, self.color_2, tol=tol)

    def save_labeled_point_cloud(self, save_repository):
        os.makedirs(save_repository, exist_ok=True)
        filename=os.path.join(save_repository, f"{self.name}")
        labels=self.get_labels()
        np.save(f"{filename}.npy",{"points":self.points, "labels":labels})
        print(f"✅ Nuage sauvegardé en .npy : {filename}")


""" classe commentée car je doute qu'elle soit utile.
class o3dpcd(PointCloud):
    def __init__(self, source):
        super().__init__(source)

    def load(self, source):
        points = np.asarray(source.points)
        colors = np.asarray(source.colors)
        return points, colors

    def get_labels(self):
        return self.infer_labels_from_colors(self.colors)
"""