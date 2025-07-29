import numpy as np
import random as rd
import trimesh
import pyvista as pv
import pyransame
import math
import os
from point_cloud import PointCloud
from generateurs.data_augmentation import DataAugmentation
from abc import ABC, abstractmethod
class Generateurs(ABC):
    def __init__(self, source_corolle: str, n:int, save_repository:str, pre_gen_augmenter:DataAugmentation, post_gen_augmenter:DataAugmentation, with_outliers:bool, normalize_points:bool=True, drop_pot:bool=True):
        self.corolle=self.trim_to_point_cloud(trimesh.load(source_corolle, force="scene"), normalize=normalize_points)
        self.n=n
        self.save_repository=save_repository
        self.pre_gen_augmenter=pre_gen_augmenter
        self.post_gen_augmenter=post_gen_augmenter
        self.with_outliers=with_outliers
        self.drop_pot=drop_pot

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

    @staticmethod
    def save_labeled_point_cloud(corolle, plante, outliers, filename="synthetic_point_cloud"):
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
        corolle = np.array(corolle)
        plante = np.array(plante)

        points_list = [corolle, plante]
        labels_list = [
            np.zeros(len(corolle), dtype=np.int64),
            np.ones(len(plante), dtype=np.int64),
        ]

        if outliers is not None and len(outliers) > 0:
            outliers = np.array(outliers)
            points_list.append(outliers)
            labels_list.append(2 * np.ones(len(outliers), dtype=np.int64))

        points_fusion = np.vstack(points_list).astype(np.float32)
        labels = np.concatenate(labels_list)

        np.save(f"{filename}.npy", {"points": points_fusion, "labels": labels})
        print(f"✅ Sauvegardé : {filename}.npy ({len(points_fusion)} points)")

        return points_fusion, labels
    
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

    @abstractmethod
    def gen_points(self):
        pass

    @abstractmethod
    def gen_dataset(self):
        pass

    @abstractmethod
    def run(self):
        pass

class GenerateurNaif(Generateurs):
    def __init__(self, source_corolle: str, n:int, save_repository:str, pre_gen_augmenter:DataAugmentation, post_gen_augmenter:DataAugmentation, with_outliers:bool, drop_pot:bool=True, max_rad_sphere=0.025, min_dens= 1000000, max_dens=10000000):
        super().__init__(source_corolle, n, save_repository, pre_gen_augmenter, post_gen_augmenter, with_outliers, drop_pot)
        self.max_rad_sphere=max_rad_sphere
        self.min_dens=min_dens
        self.max_dens=max_dens
    
    @staticmethod
    def rd_add_single_sphere(points, radius:float, density):
        points=np.array(points)
        k=rd.randint(0, len(points)-1)
        x, y, z=points[k]
        new_points=Generateurs.generate_sphere_points(x, y, z, radius, density)
        return new_points

    def gen_points(self, point_cloud):
        n_sphere=rd.randint(100, 1_000)
        new_points=[]
        for i in range(n_sphere):
            radius = min(np.abs(rd.gauss(0, self.max_rad_sphere / 5)), self.max_rad_sphere)
            dens=rd.randint(self.min_dens, self.max_dens)
            new_points.extend(self.rd_add_single_sphere(point_cloud, radius, dens))
        return(np.array(new_points))

    def gen_dataset(self):
        if self.drop_pot:
            point_cloud = self.drop(self.corolle)
        else:
            point_cloud = self.corolle

        N = point_cloud.shape[0]
        # Paramètres de déformation aléatoires pour n corolles (à intégrer dans les paramètres du générateur)
        # sigma = np.random.uniform(0.0005, 0.0015, size=(self.n, 1, 1))
        # trans = np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], size=(self.n, 1, 3))
        # angle = np.random.uniform(0, 2.2, size=(self.n, 1, 1))
        # scale_error = np.random.uniform(-0.05, 0.05, size=(self.n, 1, 1))

        pc = point_cloud[np.newaxis, :, :]
        pc = np.broadcast_to(pc, (self.n, N, 3)).copy()
        pc_deformed = self.pre_gen_augmenter.deform_tot(pc)

        # Création du dossier de sortie
        os.makedirs(self.save_repository, exist_ok=True)

        for i in range(self.n):
            point_cloud_i = np.asarray(pc_deformed[i])
            plant_points = self.gen_points(point_cloud_i)
            if self.with_outliers:
                n_outliers = np.random.randint(100, 1001)
                outlier_points = self.gen_outliers(
                    point_cloud_i, n_outliers=n_outliers
                )
                # Filtrage : on ne garde que les outliers avec y >= 0.45 (pot)
                outlier_points = [pt for pt in outlier_points if pt[1] >= 0.45]
            else:
                outlier_points = None

            corolla_dropped = self.post_gen_augmenter.deform_tot(point_cloud_i)
            plante_dropped = self.post_gen_augmenter.deform_tot(plant_points)
            outliers_dropped = self.post_gen_augmenter.deform_tot(outlier_points)
            filename = os.path.join(self.save_repository, f"sample_{i:04d}")
            self.save_labeled_point_cloud(
                corolle=corolla_dropped,
                plante=plante_dropped,
                outliers=outliers_dropped,
                filename=filename,
            )
        print(f"✅ {self.n} nuages sauvegardés dans : {self.save_repository}")

    def run(self):
        self.gen_dataset()

class GenerateurIteratif(Generateurs):
        def __init__(self, source_corolle: str, n:int, n_plants_min:int, n_plants_max:int, save_repository:str, pre_gen_augmenter:DataAugmentation, post_gen_augmenter:DataAugmentation, with_outliers:bool, min_density: float, max_density:float, min_radius: float, max_radius:float, min_decay_density:float, max_decay_density:float, min_decay_radius:float, max_decay_radius:float, max_depth: int, normalize_points=True, drop_pot:bool=True):
            super().__init__(source_corolle, n, save_repository, pre_gen_augmenter, post_gen_augmenter, with_outliers, normalize_points, drop_pot)
            self.n_plants_min=n_plants_min
            self.n_plants_max=n_plants_max
            self.min_density=min_density
            self.max_density=max_density
            self.min_radius=min_radius
            self.max_radius=max_radius
            self.min_decay_density=min_decay_density
            self.max_decay_density=max_decay_density
            self.min_decay_radius=min_decay_radius
            self.max_decay_radius=max_decay_radius
            self.max_depth

        @staticmethod
        def iterative_growth(corolle_points, x, y, z, base_density, base_radius, decay_density, decay_radius, depth_max):
            corolle_points = np.asarray(corolle_points)
            points = []
            current_density = base_density
            current_radius = base_radius
            current_center = np.array([x, y, z])
            step = 0
            yfin = float(np.max(corolle_points[:, 1]))

            # Construire le KDTree une seule fois (parce que corolle_points est fixe)
            tree = KDTree(corolle_points)

            while step < depth_max:
                sphere = generate_sphere_points(
                    *current_center, current_radius, current_density
                )
                if len(sphere) == 0:
                    break

                points.extend(sphere)

                # Trouver le point le plus haut de la sphère
                p_high = np.array(max(sphere, key=lambda pt: pt[1]))

                # Utiliser KDTree pour trouver le point le plus proche de p_high dans corolle_points
                # La méthode query renvoie une paire (distance, index)
                dist, idx_nearest = tree.query(p_high)
                new_center = corolle_points[idx_nearest]

                # Appliquer les décadences
                current_density *= decay_density
                current_radius *= decay_radius
                current_center = new_center

                step += 1

            return points

        def gen_points(self, points, base_points):
            n_plants=np.random.randint(self.n_plants_min, self.n_plants_max)
            radius=np.random.uniform(self.min_radius, self.max_radius)
            base_density=np.random.uniform(self.min_density, self.max_density)
            decay_density=np.random.uniform(self.min_decay_density, self.max_decay_density)
            decay_radius=np.random.uniform(self.min_decay_radius, self.max_decay_radius)
            grown_points = []

            for _ in range(n_plants):
                x0, y0, z0 = base_points[np.random.randint(len(base_points))]
                plant = self.iterative_growth(
                    points_corolle,
                    x0,
                    y0,
                    z0,
                    base_density,
                    radius,
                    decay_density,
                    decay_radius,
                    self.max_depth,
                )
                grown_points.extend(plant)

            return grown_points

        def gen_dataset(self):
            N=self.corolle.shape[0]
            base_points = np.array([pt for pt in point_cloud if pt[1] > 0.45 and pt[1] < 0.49])
            pc = self.corolle[np.newaxis, :, :]
            pc = np.broadcast_to(pc, (self.n, N, 3)).copy()
            pc_deformed = pre_gen_augmenter.deform_tot(pc)
            M = base_points.shape[0]
            base_points = base_points[np.newaxis, :, :]
            base_points = np.broadcast_to(base_points, (self.n, M, 3)).copy()
            base_points_deformed = pre_gen_augmenter.deform_tot(base_points)



            



"""

    @staticmethod
    def ajout_bruit_gaussien(points, n, sigma_range=0.001, random=True):
        
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
    def apply_translation(points, n, trans_range=0, random=True):
        
        Applique une translation.
        Le vecteur de translation peut être de forme (3,) (même translation pour tous les points)
        ou de forme (n, 1, 3) pour avoir une translation différente pour chaque nuage.
        
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
    def apply_random_rotation(points, n, angle_range_deg=11):
        
        Applique une rotation aléatoire autour des axes x, y et z.
        Les angles de rotation sont tirés uniformément entre -angle_range_deg et +angle_range_deg.

        Supporte les entrées de forme (N, 3) ou (n, N, 3).
        

        def rotation_matrix_xyz(angle_x, angle_y, angle_z):
            Retourne la matrice de rotation 3D composée autour des axes x, y, z.
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
    def apply_scaling(points, n, scale_error=0.05, random=True):
        
        Applique une mise à l'échelle :
        - Si `random=True`, un facteur aléatoire dans [1 - scale_error, 1 + scale_error] est tiré.
            * Si points est de forme (N, 3), une seule échelle est appliquée.
            * Si points est de forme (n, N, 3), chaque nuage reçoit une échelle différente.
        - Si `random=False`, applique un facteur fixe 1 + scale_error (scalaire ou tableau diffusable).

        Args :
            points (np.ndarray): (N, 3) ou (n, N, 3)
            scale_error (float or array): amplitude de variation (si random=True) ou facteur à appliquer.
            random (bool): active ou non la génération aléatoire.
        
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

    def deformation_tot(self, sigma_range=0.001, trans_range=np.zeros(3), angle_deg=0, scale_error=0, random_bruit=True, random_trans=True, random_scale=True):
        #if points.ndim == 2:
        #    N = points.shape[0]
        #    points = np.tile(points[None, :, :], (n_to_generate, 1, 1)) 
        #elif points.ndim == 3:
        #    if self.n != 1:
        #        print("⚠️ Attention : n_to_generate est ignoré car un batch est déjà fourni.")
        #        n_to_generate=points.shape[0]
        #else:
        #    raise ValueError("points doit être de forme (N, 3) ou (n, N, 3)")
        # Appliquer chaque transformation aléatoirement ou fixée
        points = Generateurs.ajout_bruit_gaussien(self.corolle, self.n, sigma_range=sigma_range, random=random_bruit)
        points = Generateurs.apply_translation(self.corolle, self.n, trans_range=trans_range, random=random_trans)
        points = Generateurs.apply_random_rotation(self.corolle, self.n, angle_range_deg=angle_deg)
        points = Generateurs.apply_scaling(self.corolle, self.n, scale_error=scale_error, random=random_scale)

        return points.astype(np.float32)
"""