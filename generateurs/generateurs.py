import numpy as np
import random as rd
import trimesh
import pyvista as pv
import pyransame
import math
import os
from point_cloud import PointCloud

class Generateurs:
    def __init__(self, source_corolle: np.ndarray, pc_plante:np.ndarray=None, pc_outlier: np.ndarray=None, drop_pot:bool=True, bruit:bool=True, dropout: bool=True):
        self.corolle=self.trim_to_point_cloud(trimesh.load(source_corolle, force="scene"))
        self.plante=pc_plante
        self.outliers=pc_outlier
        self.drop_pot=drop_pot
        self.bruit=bruit
        self.dropout=dropout

    @staticmethod
    def trim_to_point_cloud(mesh_scene, normalize=True):
        # extraire les meshes de mesh_scene et les fusionner
        meshes = [geom for geom in mesh_scene.geometry.values()]
        combined_mesh = trimesh.util.concatenate(meshes)

        # Convertir en mesh PyVista
        mesh_pv = pv.PolyData(
            combined_mesh.vertices,
            np.hstack(
                [np.full((len(combined_mesh.faces), 1), 3), combined_mesh.faces]
            ).astype(np.int32),
        )

        # Générer un nuage de points avec pyransame
        model_point_cloud = np.array(pyransame.surface.random_surface_points(mesh_pv, n=100_000))

        return PointCloud.normalize_points(model_point_cloud) if normalize else model_point_cloud

    @staticmethod
    def drop(points):
        return points[points[:, 1]>=0.45]

    def apply_drop_pot(self):
        self.corolle=self.drop(self.corolle)
        if self.plante is not None:
            self.plante=self.drop(self.plante)

    @staticmethod
    def ajout_bruit_gaussien(points, sigma_range=0.001, random=True):
        """
        Ajoute un bruit gaussien aux points.
        
        - Si `random=True`, un sigma aléatoire est tiré dans [0, 0.01].
            * Si points est de forme (N, 3), le même sigma est appliqué à tous les points.
            * Si points est de forme (n, N, 3), un sigma différent est appliqué à chaque nuage.
        - Si `random=False`, utilise le sigma donné (obligatoire dans ce cas).

        Args:
            points (np.ndarray): Tableau de forme (N, 3) ou (n, N, 3)
            sigma (float or array): Écart-type du bruit (ou None si random=True)
            random (bool): Active ou non la génération aléatoire

        Returns:
            np.ndarray: Points bruités
        """
        if random:
            if points.ndim == 2:
                sigma = np.random.uniform(0, sigma_range)
            elif points.ndim == 3:
                n = points.shape[0]
                sigma = np.random.uniform(0, sigma_range, size=(n, 1, 1))
            else:
                raise ValueError("points doit être de forme (N, 3) ou (n, N, 3)")
        else:
            sigma=0.0
        noise = np.random.normal(0, sigma, size=points.shape)
        return points + noise
    
    @staticmethod
    def apply_translation(points, trans_range=0, random=True):
        """
        Applique une translation.
        Le vecteur de translation peut être de forme (3,) (même translation pour tous les points)
        ou de forme (n, 1, 3) pour avoir une translation différente pour chaque nuage.
        """
        if random:
            if points.ndim == 2:
                trans = np.random.uniform(-trans_range, trans_range, size=3)
            elif points.ndim == 3:
                n = points.shape[0]
                trans = np.random.uniform(-trans_range, trans_range, size=(n, 1, 3))
            else:
                raise ValueError("points doit être de forme (N, 3) ou (n, N, 3)")
        elif not random:
            print(f"trans is None and random is False: translation is set to 0.")
            trans=np.zeros(3)
        return points + trans

    @staticmethod
    def apply_random_rotation(points, angle_range_deg=11):
        """
        Applique une rotation aléatoire autour des axes x, y et z.
        Les angles de rotation sont tirés uniformément entre -angle_range_deg et +angle_range_deg.

        Supporte les entrées de forme (N, 3) ou (n, N, 3).
        """

        def rotation_matrix_xyz(angle_x, angle_y, angle_z):
            """Retourne la matrice de rotation 3D composée autour des axes x, y, z."""
            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)],
                ]
            )
            Ry = np.array(
                [
                    [np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)],
                ]
            )
            Rz = np.array(
                [
                    [np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1],
                ]
            )
            return Rz @ Ry @ Rx  # ZYX convention

        if points.ndim == 2:
            # Un seul nuage
            if np.isscalar(angle_range_deg):
                angles = np.deg2rad(
                    np.random.uniform(-angle_range_deg, angle_range_deg, size=3)
                )
            else:
                raise ValueError(
                    "angle_range_deg doit être un scalaire pour un nuage unique."
                )
            R = rotation_matrix_xyz(*angles)
            return points @ R.T

        elif points.ndim == 3:
            n = points.shape[0]
            if np.isscalar(angle_range_deg):
                angles = np.deg2rad(
                    np.random.uniform(-angle_range_deg, angle_range_deg, size=(n, 3))
                )
            else:
                angle_range_deg = np.asarray(angle_range_deg).reshape(-1, 1)
                angles = np.deg2rad(
                    np.random.uniform(-angle_range_deg, angle_range_deg, size=(n, 3))
                )  # shape (n, 3)

            R = np.array(
                [rotation_matrix_xyz(ax, ay, az) for ax, ay, az in angles]
            )  # (n, 3, 3)
            return np.matmul(points, np.transpose(R, (0, 2, 1)))

        else:
            raise ValueError("La dimension de 'points' doit être 2 ou 3.")

    @staticmethod
    def apply_scaling(points, scale_error=0.05, random=True):
        """
        Applique une mise à l'échelle :
        - Si `random=True`, un facteur aléatoire dans [1 - scale_error, 1 + scale_error] est tiré.
            * Si points est de forme (N, 3), une seule échelle est appliquée.
            * Si points est de forme (n, N, 3), chaque nuage reçoit une échelle différente.
        - Si `random=False`, applique un facteur fixe 1 + scale_error (scalaire ou tableau diffusable).

        Args :
            points (np.ndarray): (N, 3) ou (n, N, 3)
            scale_error (float or array): amplitude de variation (si random=True) ou facteur à appliquer.
            random (bool): active ou non la génération aléatoire.
        """
        if random:
            if points.ndim == 2:
                scale_factor = np.random.uniform(1 - scale_error, 1 + scale_error)
            elif points.ndim == 3:
                n = points.shape[0]
                scale_factor = np.random.uniform(1 - scale_error, 1 + scale_error, size=(n, 1, 1))
            else:
                raise ValueError("points doit être de forme (N, 3) ou (n, N, 3)")
        else:
            scale_factor = 1 + scale_error 
        return points * scale_factor

    @staticmethod
    def deformation_tot(points, n_to_generate=1, sigma_range=0.001, trans_range=np.zeros(3), angle_deg=0, scale_error=0, random_bruit=True, random_trans=True, random_scale=True):
        if points.ndim == 2:
            N = points.shape[0]
            points = np.tile(points[None, :, :], (n_to_generate, 1, 1)) 
        elif points.ndim == 3:
            if n_to_generate != 1:
                print("⚠️ Attention : n_to_generate est ignoré car un batch est déjà fourni.")
                n_to_generate=points.shape[0]
        else:
            raise ValueError("points doit être de forme (N, 3) ou (n, N, 3)")
        # Appliquer chaque transformation aléatoirement ou fixée
        points = Generateurs.ajout_bruit_gaussien(points, sigma_range=sigma_range, random=random_bruit)
        points = Generateurs.apply_translation(points, trans_range=trans_range, random=random_trans)
        points = Generateurs.apply_random_rotation(points, angle_range_deg=angle_deg)
        points = Generateurs.apply_scaling(points, scale_error=scale_error, random=random_scale)

        return points.astype(np.float32)

    @staticmethod
    def generate_sphere_points(x, y, z, radius, density, y_min=0.4):
        # Calculer le nombre de points à générer
        n_points = max(1, int(density * math.pi * (radius**3)))

        # Génération vectorisée des angles et des rayons :
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        # on prend sa valeur absolue et on la borne par le rayon maximum
        r = np.abs(np.random.normal(0, radius, n_points))
        r = np.minimum(r, radius)

        # Calcul vectorisé des coordonnées
        x_gen = x + r * np.sin(phi) * np.cos(theta)
        y_gen = y + r * np.sin(phi) * np.sin(theta)
        z_gen = z + r * np.cos(phi)

        # Filtrer les points qui respectent la condition y >= y_min
        mask = y_gen >= y_min
        points = np.stack([x_gen[mask], y_gen[mask], z_gen[mask]], axis=1)

        return points.tolist()

    def save_labeled_point_cloud(self, filename="synthetic_point_cloud"):
        """
        Fusionne les nuages de points (corolle, plante, outliers),
        crée des labels associés et sauvegarde le tout dans un fichier .npy unique.

        Format de sauvegarde : fichier .npy contenant un dictionnaire :
            {
                'points': (N, 3) array of float32,
                'labels': (N,) array of int64
            }

        Labels :
            - Corolle : 0
            - Plante  : 1
            - Outlier : 2 (si fourni)
        """
        self.corolle = np.array(self.corolle)
        self.plante = np.array(self.plante)

        points_list = [self.corolle, self.plante]
        labels_list = [
            np.zeros(len(self.corolle), dtype=np.int64),
            np.ones(len(self.plante), dtype=np.int64),
        ]

        if self.outliers is not None and len(self.outliers) > 0:
            self.outliers = np.array(self.outliers)
            points_list.append(self.outliers)
            labels_list.append(2 * np.ones(len(self.outliers), dtype=np.int64))

        points_fusion = np.vstack(points_list).astype(np.float32)
        labels = np.concatenate(labels_list)

        np.save(f"{filename}.npy", {"points": points_fusion, "labels": labels})
        print(f"✅ Sauvegardé : {filename}.npy ({len(points_fusion)} points)")

        return points_fusion, labels

    @staticmethod
    def dropout_points(points, mean_percent=24, std_percent=8, max_percent=60):
        """
        Supprime aléatoirement une proportion de points dans un nuage de points,
        avec un taux de dropout tiré d'une loi normale tronquée.

        Args:
            points (np.ndarray): Tableau (N, 3) de points 3D.
            mean_percent (float): Moyenne de la loi normale du taux de dropout.
            std_percent (float): Écart-type de la loi normale.
            max_percent (float): Taux maximum de dropout (en %).

        Returns:
            np.ndarray: Nuage de points après dropout.
        """
        if points is None:
            return None

        points = np.asarray(points)
        n_points = len(points)
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
    
    def dropout_pc(self, mean_percent=24, std_percent=8, max_percent=60):
        self.corolle=Generateurs.dropout_points(self.corolle, mean_percent=mean_percent, std_percent=std_percent, max_percent=max_percent)
        self.plante=Generateurs.dropout_points(self.plante, mean_percent=mean_percent, std_percent=std_percent, max_percent=max_percent)
        self.outliers=Generateurs.dropout_points(self.outliers, mean_percent=mean_percent, std_percent=std_percent, max_percent=max_percent)

    @staticmethod
    def gen_outliers(points, n_outliers=1000, lateral_std=0.05, vertical_std=0.02, y_jitter_centered=True, min_outliers=10):
        points=np.asarray(points)
        if points.ndim==2:
            points=points[np.newaxis, :, :]

        n, N, _ =points.shape
        valid_mask = points[:, :, 1] >= 0.45
        valid_counts = valid_mask.sum(axis=1)

        k = n_outliers
        while np.any(valid_counts < k):
                k //= 2
                if k < min_outliers:
                    raise ValueError(
                        f"Impossible de générer au moins {min_outliers} outliers avec y ≥ 0.45 pour toutes les corolles."
                    )

        rand = np.random.rand(n, N)
        rand[~valid_mask] = np.inf
        topk_indices = np.argpartition(rand, k, axis=1)[:, :k]
        base_points = np.take_along_axis(points, topk_indices[:, :, None], axis=1)

        offset_xz = np.random.normal(scale=lateral_std, size=(n, k, 2))
        offset_y = (np.random.normal(scale=vertical_std, size=(n, k, 1)) if y_jitter_centered
                    else -np.abs(np.random.normal(scale=vertical_std, size=(n, k, 1))))

        offsets = np.zeros_like(base_points)
        offsets[..., 0] = offset_xz[..., 0]
        offsets[..., 1] = offset_y[..., 0]
        offsets[..., 2] = offset_xz[..., 1]

        outliers = base_points + offsets
        return outliers.squeeze(0) if outliers.shape[0] == 1 else outliers


class Generateur_naif(Generateurs):
    def __init__(self, source_corolle: np.ndarray, pc_plante:np.ndarray=None, pc_outlier: np.ndarray=None, drop_pot:bool=True, bruit:bool=True, dropout: bool=True, max_rad_sphere=0.025, min_dens= 1000000, max_dens=10000000):
        super.__init__(self, source_corolle: np.ndarray, pc_plante:np.ndarray=None, pc_outlier: np.ndarray=None, drop_pot:bool=True, bruit:bool=True, dropout: bool=True)
        self.max_rad_sphere=max_rad_sphere
        self.min_dens=min_dens
        self.max_dens=max_dens
    def rd_add_single_sphere()