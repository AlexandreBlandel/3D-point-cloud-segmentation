import numpy as np
import pyvista as pv
import trimesh
import pyransame
import open3d as o3d


"""
1- normalisation
"""


def normalisation(point_cloud):
    if isinstance(point_cloud, np.ndarray):
        # Étape 1 : Centrage autour du barycentre
        centroid = np.mean(point_cloud, axis=0)
        points_centered = point_cloud - centroid

        # Étape 2 : Mise à l’échelle (normalisation par la plus grande norme)
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        points_normalized = points_centered / max_dist

        # Étape 3 : Mise à l’échelle dans le cube [0, 1]^3
        point_cloud = (points_normalized + 1) / 2

        # Étape 4 : Renormalisation dans [0, 1]^3
        min_vals = np.min(point_cloud, axis=0)
        max_vals = np.max(point_cloud, axis=0)
        point_cloud = (point_cloud - min_vals) / (max_vals - min_vals)

        return point_cloud

    elif isinstance(point_cloud, o3d.geometry.PointCloud):
        new_points = normalisation(np.asarray(point_cloud.points))
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)
        if point_cloud.has_colors():
            new_pcd.colors = point_cloud.colors
        return new_pcd


"""
2- échantillonnage du modèle trimesh de la corolle
"""


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

    return normalisation(point_cloud_dense)


"""
3- sous-échantillonnage pour éviter les problèmes mémoires:
"""


def downsample_pcd(pcd, ratio=0.05, max_points=25_000):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    n = len(points)
    m = int(min(ratio * n, max_points))
    idx = np.random.choice(n, m, replace=False)
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(points[idx])
    sampled_pcd.colors = o3d.utility.Vector3dVector(colors[idx])

    return sampled_pcd


"""
4- rotation des nuages de points autour de l'axe x
"""


def rotate_pcd(pcd, axis="x", degrees=-90, center=(0, 0, 0)):
    """
    Applique une rotation autour de l'axe x au point cloud donné.
    Par défaut, fait passer l'axe Y en Z (rotation de +90°).
    """
    radians = np.deg2rad(degrees)
    if axis == "x":
        R = pcd.get_rotation_matrix_from_xyz((radians, 0, 0))
    elif axis == "y":
        R = pcd.get_rotation_matrix_from_xyz((0, radians, 0))
    elif axis == "z":
        R = pcd.get_rotation_matrix_from_xyz((0, 0, radians))
    else:
        raise ValueError("Axe invalide. Choisir parmi 'x', 'y' ou 'z'.")

    pcd.rotate(R, center=center)
    return pcd
