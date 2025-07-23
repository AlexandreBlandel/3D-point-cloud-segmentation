import numpy as np
import open3d as o3d

"""
def visualiser_sample_open3d(filepath):
    
    #Charge et affiche un fichier .npy contenant un dictionnaire
    #{'points': (N, 3), 'labels': (N,)} avec Open3D, avec couleurs par classe :
    #- Corolle (label=0) : gris
    #- Plante  (label=1) : vert
    #- Outlier (label=2) : rouge
    
    data = np.load(filepath, allow_pickle=True).item()
    points = data["points"]
    labels = data["labels"]

    # Créer un nuage de points Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Assigner des couleurs par classe
    colors = np.zeros_like(points)
    colors[labels == 0] = [0.6, 0.6, 0.6]  # gris pour corolle
    colors[labels == 1] = [0.0, 0.8, 0.0]  # vert pour plante
    colors[labels == 2] = [1.0, 0.0, 0.0]  # rouge pour outliers

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Afficher
    o3d.visualization.draw_geometries([pcd])
"""


def visualiser_sample_open3d_robuste(source, return_pc=True):
    """
    Visualise un nuage de points avec Open3D.

    Paramètres
    ----------
    source : str ou dict
        - Chemin vers un fichier `.npy` contenant {'points': (N, 3), 'labels': (N,)}
        - Chemin vers un fichier `.ply`
        - Ou dictionnaire {'points': (N, 3), 'labels': (N,)} déjà chargé
        - ou nuage de point o3d

    Couleurs utilisées :
        - label=0 : gris (corolle)
        - label=1 : vert (plante)
        - label=2 : rouge (outlier)
        - Si pas de label, on garde les couleurs du .ply ou du bleu par défaut
    """

    # === Cas 1 : fichier .npy
    if isinstance(source, str) and source.endswith(".npy"):
        data = np.load(source, allow_pickle=True).item()
        points = data["points"]
        labels = data["labels"]
        has_labels = True

    # === Cas 2 : fichier .ply
    elif isinstance(source, str) and source.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(source)
        points = np.asarray(pcd.points)
        labels = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = np.full_like(points, fill_value=[0.2, 0.5, 1.0])  # bleu par défaut
        has_labels = False

    # === Cas 3 : dict déjà chargé
    elif isinstance(source, dict):
        points = source["points"]
        labels = source["labels"]
        has_labels = True

    elif isinstance(source, o3d.geometry.PointCloud):
        o3d.visualization.draw_geometries([source])
        if return_pc:
            return source

    else:
        raise ValueError(
            "Le paramètre 'source' doit être un chemin .npy, .ply ou un dict."
        )

    # Créer un nuage de points Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if has_labels:
        colors = np.zeros_like(points)
        colors[labels == 0] = [0.6, 0.6, 0.6]  # gris pour corolle
        colors[labels == 1] = [0.0, 0.8, 0.0]  # vert pour plante
        colors[labels == 2] = [1.0, 0.0, 0.0]  # rouge pour outliers
    # Sinon, `colors` a déjà été défini depuis le .ply ou par défaut

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    if return_pc:
        return pcd
