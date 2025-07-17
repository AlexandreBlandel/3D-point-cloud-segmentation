import numpy as np
import pyvista as pv
import trimesh
import pyransame

"""
1- échantillonnage du modèle trimesh de la corolle
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
