import trimesh
import numpy as np
import os
from scipy.spatial import KDTree
from utils import trim_to_pointcloud
from utils_gen import deformation_tot
from utils_gen import save_labeled_point_cloud
from utils_gen import dropout_pointcloud
from utils_gen import generate_sphere_points
from utils_gen import enlever_pot
from outlier import generate_outliers_batch_v3


def iterative_growth(
    corolle_points,
    x,
    y,
    z,
    base_density,
    base_radius,
    decay_density,
    decay_radius,
    depth_max=20,
):
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


def grow_plants_iterative(points_corolle, base_points, n_plants=10, **kwargs):
    # print(len(base_points))
    # ys = [pt[1] for pt in points_corolle]
    # print(f"min y: {min(ys)}, max y: {max(ys)}")

    grown_points = []

    for _ in range(n_plants):
        x0, y0, z0 = base_points[np.random.randint(len(base_points))]
        plant = iterative_growth(
            points_corolle,
            x0,
            y0,
            z0,
            kwargs["base_density"],
            kwargs["radius"],
            kwargs["decay_density"],
            kwargs["decay_radius"],
            kwargs["max_depth"],
        )
        grown_points.extend(plant)

    return grown_points


def gen_dataset(
    n,
    scene_link="Corolle_L_pour_Canoscan_NonRenf_Filins_MESH.obj",
    with_outliers=False,
    save_dir="generateurs/dataset_iter_outlier_dropout",
):

    # Étape 1 : Chargement du mesh et extraction du nuage
    mesh_scene = trimesh.load(scene_link, force="scene")
    point_cloud = np.array(trim_to_pointcloud(mesh_scene))  # shape (N, 3)
    N = point_cloud.shape[0]

    # on repère les points candidats pour servir de base de la pousse de plante avant erreur de translation, de rotation etc
    base_points = np.array([pt for pt in point_cloud if pt[1] > 0.45 and pt[1] < 0.49])

    # Paramètres de déformation aléatoires pour n corolles
    sigma = np.random.uniform(0.0005, 0.0015, size=(n, 1, 1))
    trans = np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], size=(n, 1, 3))
    angle = np.random.uniform(0, 2.2, size=(n, 1, 1))
    scale_error = np.random.uniform(-0.05, 0.05, size=(n, 1, 1))

    pc = point_cloud[np.newaxis, :, :]
    pc = np.broadcast_to(pc, (n, N, 3)).copy()
    pc_deformed = deformation_tot(pc, sigma, trans, angle, scale_error)
    M = base_points.shape[0]
    base_points = base_points[np.newaxis, :, :]
    base_points = np.broadcast_to(base_points, (n, M, 3)).copy()
    base_points_deformed = deformation_tot(
        base_points, sigma, trans, angle, scale_error
    )

    # Paramètres de génération de plantes
    central_radius = 0.025
    central_base_density = 4_000_000
    central_decay_density = 0.86
    central_decay_radius = 0.988
    std_radius = 0.0041
    min_val = 0.95
    max_val = 0.99925
    central_max_depth = 250

    # Création du dossier de sortie
    os.makedirs(save_dir, exist_ok=True)

    # Boucle pour outliers, plantes, sauvegarde
    for i in range(n):
        corolle_points = pc_deformed[i]
        base_points_i = base_points_deformed[i]

        # Génération plante
        n_plants = np.random.randint(12, 35)
        radius = np.random.uniform(0.8 * central_radius, 1.25 * central_radius)
        base_density = np.random.uniform(
            0.85 * central_base_density, 1.25 * central_base_density
        )
        decay_density = np.random.uniform(
            0.85 * central_decay_density, 1.014 * central_decay_density
        )
        decay_radius = np.clip(
            np.random.normal(loc=central_decay_radius, scale=std_radius),
            min_val,
            max_val,
        )
        # decay_radius = np.random.uniform(0.978 * central_decay_radius, 1.01 * central_decay_radius)

        plant_points = grow_plants_iterative(
            corolle_points.tolist(),
            base_points_i,
            n_plants=n_plants,
            base_density=base_density,
            radius=radius,
            decay_density=decay_density,
            decay_radius=decay_radius,
            max_depth=central_max_depth,
        )

        # Génération outliers
        if with_outliers:
            n_outliers = np.random.randint(250, 2501)
            outlier_points = generate_outliers_batch_v3(
                corolle_points, n_outliers=n_outliers
            )

            # Filtrage : on ne garde que les outliers avec y >= 0.45
            outlier_points = [pt for pt in outlier_points if pt[1] >= 0.45]

            if len(outlier_points) == 0:
                outlier_points = None  # pour éviter un array vide dans vstack
        else:
            outlier_points = None

        print(f"[{i}] Avant nettoyage :")
        print(f"  corolle_points: {len(corolle_points)}")
        print(f"  plant_points: {len(plant_points)}")
        print(
            f"  outlier_points: {0 if outlier_points is None else len(outlier_points)}"
        )

        corolle_points = enlever_pot(corolle_points)
        print(f"[{i}] Après enlever_pot : {len(corolle_points)}")

        corolle_points = dropout_pointcloud(corolle_points)
        print(f"[{i}] Après dropout corolle : {len(corolle_points)}")

        plant_points = dropout_pointcloud(plant_points)
        print(f"[{i}] Après dropout plante : {len(plant_points)}")

        outlier_points = dropout_pointcloud(outlier_points)
        print(
            f"[{i}] Après dropout outliers : {0 if outlier_points is None else len(outlier_points)}"
        )

        # Sauvegarde fusionnée
        filename = os.path.join(save_dir, f"sample_{i:04d}")
        save_labeled_point_cloud(
            point_corolla=corolle_points,
            point_plant=plant_points,
            point_outlier=outlier_points,
            filename=filename,
        )

    print(f"✅ {n} nuages sauvegardés dans : {save_dir}")
