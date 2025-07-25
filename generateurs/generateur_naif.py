import numpy as np
import random as rd
import trimesh
import os
from utils import trim_to_pointcloud
from generateurs.utils_gen import deformation_tot
from generateurs.utils_gen import save_labeled_point_cloud
from generateurs.utils_gen import dropout_pointcloud
from generateurs.utils_gen import generate_sphere_points
from generateurs.outlier import generate_outliers_batch_v3
from generateurs.utils_gen import enlever_pot

# Définition de paramètres:
max_rad_sphere = 0.025
min_dens = 1000000
max_dens = 10000000


def rd_add_single_sphere(point_cloud_corolla, radius: float, density):
    """
    Ajoute une sphère de points à un endroit au hasard
    Entrées:
        points: nuage de point d'entrée ie le nuage de points de la corolle
        radius: rayon de la sphere
        density: densité de la sphere
    Sortie:
        new_points: les points plantes d'une sphère
    """
    # Choix du point autour duquel on va créer une sphère
    point_cloud_corolla = np.array(point_cloud_corolla)
    x, y, z = 0, 0, 0
    k = rd.randint(0, len(point_cloud_corolla) - 1)
    x, y, z = point_cloud_corolla[k]

    # Création des points générés artificiellement
    new_points = generate_sphere_points(x, y, z, radius, density)
    # for i in range(n_points):
    # calcul des coordonnées du point
    # theta = rd.uniform(0, 2 * math.pi)
    # phi = rd.uniform(0, math.pi)
    # r = min(np.abs(rd.gauss(0, radius)), radius)
    # x_gen = x + r * math.sin(phi) * math.cos(theta)
    # y_gen = y + r * math.sin(phi) * math.sin(theta)
    # z_gen = z + r * math.cos(phi)
    # new_points.append([x_gen, y_gen, z_gen])
    # np.concatenate((points, np.array([[x_gen, y_gen, z_gen]])), axis=0)

    return new_points


def gen_points(
    point_cloud_corolla, max_rad=max_rad_sphere, min_dens=min_dens, max_dens=max_dens
):
    n_sphere = int(100000 * rd.uniform(0.05, 0.3))
    new_points = []
    for i in range(n_sphere):
        radius = min(np.abs(rd.gauss(0, max_rad / 5)), max_rad)
        dens = rd.randint(min_dens, max_dens)
        new_points.extend(rd_add_single_sphere(point_cloud_corolla, radius, dens))
    return new_points


def gen_dataset_naif(
    n,
    with_outliers=False,
    corolla_link="Corolle_L_pour_Canoscan_NonRenf_Filins_MESH.obj",
    save_dir="data/synthetiques/dataset_naif",
    max_rad=max_rad_sphere,
    min_dens=min_dens,
    max_dens=max_dens,
):
    # Étape 1 : Chargement du mesh et extraction du nuage
    mesh_scene = trimesh.load(corolla_link, force="scene")
    point_cloud = np.array(trim_to_pointcloud(mesh_scene))  # shape (N, 3)
    point_cloud = enlever_pot(point_cloud)
    N = point_cloud.shape[0]

    # Paramètres de déformation aléatoires pour n corolles
    sigma = np.random.uniform(0.0005, 0.0015, size=(n, 1, 1))
    trans = np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], size=(n, 1, 3))
    angle = np.random.uniform(0, 2.2, size=(n, 1, 1))
    scale_error = np.random.uniform(-0.05, 0.05, size=(n, 1, 1))

    pc = point_cloud[np.newaxis, :, :]
    pc = np.broadcast_to(pc, (n, N, 3)).copy()
    pc_deformed = deformation_tot(pc, sigma, trans, angle, scale_error)

    # Création du dossier de sortie
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n):
        point_cloud_i = pc_deformed[i]
        plant_points = gen_points(point_cloud_i, max_rad, min_dens, max_dens)
        if with_outliers:
            n_outliers = np.random.randint(100, 1001)
            outlier_points = generate_outliers_batch_v3(
                point_cloud_i, n_outliers=n_outliers
            )
            # Filtrage : on ne garde que les outliers avec y >= 0.45 (pot)
            outlier_points = [pt for pt in outlier_points if pt[1] >= 0.45]
        else:
            outlier_points = None

        corolla_dropped = dropout_pointcloud(point_cloud_i)
        plante_dropped = dropout_pointcloud(plant_points)
        outliers_dropped = dropout_pointcloud(outlier_points)
        filename = os.path.join(save_dir, f"sample_{i:04d}")
        save_labeled_point_cloud(
            point_corolla=corolla_dropped,
            point_plant=plante_dropped,
            point_outlier=outliers_dropped,
            filename=filename,
        )
    print(f"✅ {n} nuages sauvegardés dans : {save_dir}")
