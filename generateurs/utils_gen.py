"""
Fichier comportant tous les outils communs aux générateurs!
1- échantillonnage du modèle trimesh de la corolle: transformation du modèle trimesh de la corolle en nuage de points: trim_to_pointcloud
2- ajout de bruit sur le nuage de point de la corolle
3- variation de la densité en enlevant des points au hasard: dropout_pointcloud
4- Génération de la brique de base de nos générateurs: points dans une sphère
5- labellisation des points et enregistrement du nuage de points: save_labeled_pointcloud
"""

import trimesh
import pyvista as pv
import pyransame
import numpy as np
import math


def trim_to_pointcloud(mesh_scene):
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
    point_cloud_dense = pyransame.surface.random_surface_points(mesh_pv, n=100_000)

    # Étape 1 : Centrage autour du barycentre
    centroid = np.mean(point_cloud_dense, axis=0)
    points_centered = point_cloud_dense - centroid

    # Étape 2 : Mise à l’échelle (normalisation par la plus grande norme)
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / max_dist

    # Étape 3 : Mise à l’échelle dans le cube [0, 1]^3
    point_cloud = (points_normalized + 1) / 2

    # Étape 4 : Suppression des points correspondant au pot de la corolle (y < 0.45)
    # point_cloud = point_cloud[point_cloud[:, 1] >= 0.45]
    # cette étape doit être faite plus tard pour pouvoir choisir les points de base de la pousse des plantes dans le cas itératif.
    # On le remplace par une fonction enlever_pot qu'on utilisera au bon endroit en fonction du générateur

    # Étape 5 : Renormalisation dans [0, 1]^3
    min_vals = np.min(point_cloud, axis=0)
    max_vals = np.max(point_cloud, axis=0)
    point_cloud = (point_cloud - min_vals) / (max_vals - min_vals)

    return point_cloud


def enlever_pot(points_corolle):
    return points_corolle[points_corolle[:, 1] >= 0.45]


"""
2- ajout de déformation sur le nuage de point de la corolle
"""


def ajout_bruit_gaussien(points, sigma=0.01):
    """
    Ajoute un bruit gaussien aux points.
    Peut fonctionner sur un tableau de forme (N, 3) ou (n, N, 3).
    """
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise


def apply_translation(points, trans=np.array([0.05, -0.03, 0.02])):
    """
    Applique une translation.
    Le vecteur de translation peut être de forme (3,) (même translation pour tous les points)
    ou de forme (n, 1, 3) pour avoir une translation différente pour chaque nuage.
    """
    return points + trans


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


def apply_rotation(points, angle_deg=11):
    """
    Applique une rotation autour de l'axe z.
    Cette fonction supporte les entrées de forme (N, 3) ou (n, N, 3).

    Si angle_deg est un scalaire, la même rotation est appliquée à tous les points.
    Si angle_deg est un array (de forme (n,) ou similaire), alors une rotation différente
    sera appliquée à chaque nuage.
    """
    if points.ndim == 2:
        # Cas d'un unique nuage, shape (N, 3)
        theta = np.deg2rad(angle_deg)
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        return points @ R.T
    elif points.ndim == 3:
        # Cas de n nuages, shape (n, N, 3)
        if np.isscalar(angle_deg):
            # Même angle pour tous les nuages
            theta = np.deg2rad(angle_deg)
            R = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            # Broadcasting suffisant : R de shape (3, 3) appliqué à chaque nuage
            return points @ R.T
        else:
            # angle_deg est spécifique à chaque nuage, par exemple de forme (n,)
            angle_deg = np.asarray(angle_deg).reshape(
                -1
            )  # s'assurer d'avoir un vecteur de forme (n,)
            theta = np.deg2rad(angle_deg)  # angle par nuage
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            n_batches = theta.shape[0]
            # Construire pour chaque nuage sa matrice de rotation : shape (n, 3, 3)
            R = np.zeros((n_batches, 3, 3))
            R[:, 0, 0] = cos_theta
            R[:, 0, 1] = -sin_theta
            R[:, 1, 0] = sin_theta
            R[:, 1, 1] = cos_theta
            R[:, 2, 2] = 1
            # Appliquer la rotation pour chaque nuage :
            # Pour chaque b, new_points[b] = points[b] @ R[b].T
            return np.matmul(points, np.transpose(R, (0, 2, 1)))
    else:
        raise ValueError("La dimension de 'points' doit être 2 ou 3.")


def apply_scaling(points, scale_error=0.05):
    """
    Applique une mise à l'échelle.
    Le paramètre scale_error peut être un scalaire ou un tableau diffusable aux dimensions
    de points (par exemple, de forme (n,1,1) pour n nuages).
    """
    scale_factor = 1 + scale_error
    return points * scale_factor


# Paramètres de déformation aléatoires pour n corolles
# sigma = np.random.uniform(0.0005, 0.0015)
# trans = np.random.uniform([-1, -1, -1], [1, 1, 1])
# angle = np.random.uniform(0, 11)
# scale_error = np.random.uniform(-0.4, 0.4)


def deformation_tot(
    points,
    sigma=0.001,
    trans=np.array([0, 0, 0]),
    angle_deg=0,
    scale_error=0,
):
    """
    Applique successivement :
      - du bruit gaussien,
      - une translation,
      - une rotation autour de l'axe z,
      - une mise à l'échelle.

    Peut traiter un unique nuage de forme (N, 3) ou plusieurs nuages de forme (n, N, 3)
    en fonction des paramètres (sigma, trans, angle_deg, scale_error) qui peuvent être scalaires
    ou arrays diffusables.
    """
    points = ajout_bruit_gaussien(points, sigma)
    points = apply_translation(points, trans)
    # Corriger le format de angle_deg si nécessaire
    if isinstance(angle_deg, np.ndarray) and angle_deg.ndim > 1:
        angle_deg = angle_deg.reshape(angle_deg.shape[0])  # (n, 1, 1) → (n,)
    points = apply_random_rotation(points, angle_deg)
    points = apply_scaling(points, scale_error)
    return points.astype(np.float32)


"""
3- Variation de la densité
"""


def dropout_pointcloud(points, mean_percent=24, std_percent=8, max_percent=60):
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


"""
4- Génération de la brique de base de nos générateurs: points dans une sphère 
"""


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


"""
5- Labellisation et sauvegarde
"""


def save_labeled_point_cloud(
    point_corolla, point_plant, point_outlier=None, filename="synthetic_point_cloud"
):
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
    point_corolla = np.array(point_corolla)
    point_plant = np.array(point_plant)

    points_list = [point_corolla, point_plant]
    labels_list = [
        np.zeros(len(point_corolla), dtype=np.int64),
        np.ones(len(point_plant), dtype=np.int64),
    ]

    if point_outlier is not None and len(point_outlier) > 0:
        point_outlier = np.array(point_outlier)
        points_list.append(point_outlier)
        labels_list.append(2 * np.ones(len(point_outlier), dtype=np.int64))

    points_fusion = np.vstack(points_list).astype(np.float32)
    labels = np.concatenate(labels_list)

    np.save(f"{filename}.npy", {"points": points_fusion, "labels": labels})
    print(f"✅ Sauvegardé : {filename}.npy ({len(points_fusion)} points)")

    return points_fusion, labels
