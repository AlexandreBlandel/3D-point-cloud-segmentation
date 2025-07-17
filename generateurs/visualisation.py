import numpy as np
import open3d as o3d


def visualiser_sample_open3d(filepath):
    """
    Charge et affiche un fichier .npy contenant un dictionnaire
    {'points': (N, 3), 'labels': (N,)} avec Open3D, avec couleurs par classe :
    - Corolle (label=0) : gris
    - Plante  (label=1) : vert
    - Outlier (label=2) : rouge
    """
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
