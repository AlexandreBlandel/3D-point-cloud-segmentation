import numpy as np
import pyvista as pv
import trimesh
import pyransame

"""
1- échantillonnage du modèle trimesh de la corolle
"""


def normalisation(point_cloud):
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
