from metrics.metrics import Distance
from metrics.histograms import PointCloudHistograms
from utils import normalisation
from compare.visualisation import visualiser_sample_open3d_robuste
from utils import downsample_pcd
from utils import rotate_pcd
import numpy as np
from metrics.histogram_distance import HistogramDistance


def compare_pc(
    pc_path,
    ref_pc_path="Chalon_en_Champagne_1_candidates_pointcloud_keep.ply",
    metrics=["earth_mover", "sinkhorn", "schamfer"],
    indicators_to_plot=["distance_to_y_axis", "nearest_distance", "y_distribution"],
):

    pc = visualiser_sample_open3d_robuste(pc_path, return_pc=True)
    ref_pc = visualiser_sample_open3d_robuste(ref_pc_path, return_pc=True)
    ref_pc = rotate_pcd(ref_pc, degrees=-90)
    # On vérifie que la rotation s'est bien passée
    # ref_pc = visualiser_sample_open3d_robuste(ref_pc, return_pc=True)
    pc = downsample_pcd(pc)
    ref_pc = downsample_pcd(ref_pc)

    # On vérifie si le down_sampling se fait bien
    # pc = visualiser_sample_open3d_robuste(pc)
    # ref_pc = visualiser_sample_open3d_robuste(ref_pc)
    pc = normalisation(pc)
    ref_pc = normalisation(ref_pc)

    # On vérifie que la normalisation s'est bien passée:
    pc = visualiser_sample_open3d_robuste(pc, return_pc=True)
    ref_pc = visualiser_sample_open3d_robuste(ref_pc, return_pc=True)

    points = np.asarray(pc.points)
    ref_points = np.asarray(ref_pc.points)

    # === Calcul des métriques de distance
    print(" Distances entre les deux nuages :")
    distance_obj = Distance(normalize=False)  # On a déjà normalisé

    for metric in metrics:
        method = getattr(distance_obj, metric)
        try:
            value = method(points, ref_points)
            print(f" {metric} : {value:.4f}")
        except Exception as e:
            print(f" Erreur lors du calcul de {metric} : {e}")

    # === Histogrammes des indicateurs sur le 1er nuage
    print("\n Histogrammes des indicateurs :")
    pc_hist = PointCloudHistograms(points)
    ref_pc_hist = PointCloudHistograms(ref_points)
    dist = HistogramDistance()
    for indicator in indicators_to_plot:
        try:
            print(f"\n▶ Indicateur : {indicator}")
            h1 = pc_hist.compute_histogram(indicator)
            h2 = ref_pc_hist.compute_histogram(indicator)

            h1_counts, h1_edges = h1
            h2_counts, h2_edges = h2

            pc_hist.plot_histogram(indicator, hist_data=(h1_counts, h1_edges))
            ref_pc_hist.plot_histogram(indicator, hist_data=(h2_counts, h2_edges))

            print(f"  L1         : {dist.l1(h1_counts, h2_counts):.4f}")
            print(f"  L2         : {dist.l2(h1_counts, h2_counts):.4f}")
            print(f"  Chi²       : {dist.chi_square(h1_counts, h2_counts):.4f}")
            print(f"  KL         : {dist.kl_divergence(h1_counts, h2_counts):.4f}")
            print(f"  Jensen-Shannon : {dist.js_divergence(h1_counts, h2_counts):.4f}")

        except Exception as e:
            print(f" Erreur avec l'indicateur '{indicator}' : {e}")
