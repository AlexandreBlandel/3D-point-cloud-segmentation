from generateurs.generateur_iteratif import gen_dataset
from generateurs.generateur_naif import gen_dataset_naif
from compare.visualisation import visualiser_sample_open3d_robuste
from compare.compare import compare_pc
from metrics.indicators import (
    Indicator,
    NearestNeighborDistanceIndicator,
    DistanceToYAxisIndicator,
    YCoordinateDistributionIndicator,
)
from metrics.histogram_distance import HistogramDistance
from metrics.metrics import Distance
from point_cloud import PointCloud, npypcd, plypcd, o3dpcd, dictpcd

if __name__ == "__main__":
    # n = 1
    # dataset = gen_dataset(n, with_outliers=True)
    # compare_pc( pc_path="data/synthetiques/dataset_iter_outlier_dropout/sample_0000.npy", ref_pc_path="data/reel/Chalon_en_Champagne_1_candidates_pointcloud_keep.ply", metrics=["earth_mover", "sinkhorn", "schamfer"], indicators_to_plot=["distance_to_y_axis", "nearest_distance", "y_distribution"])
    pc = npypcd("data/synthetiques/dataset_iter_outlier_dropout/sample_0000.npy")
    pc_ref = plypcd("data/reel/Chalon_en_Champagne_1_candidates_pointcloud_keep.ply")
    metrics = ["earth_mover", "sinkhorn", "schamfer"]
    indicators_to_plot = ["distance_to_y_axis", "nearest_distance", "y_distribution"]
    indicator_map = {
        "nearest_distance": NearestNeighborDistanceIndicator,
        "distance_to_y_axis": DistanceToYAxisIndicator,
        "y_distribution": YCoordinateDistributionIndicator,
    }
    hist_dist = HistogramDistance()

    # Visualisations et Histogrammes sans normalisation des points mais avec normalisation des histogrammes
    pc.visualiser(return_pc=False)
    pc_ref.visualiser(return_pc=False)
    for indicator_name in indicators_to_plot:
        indicator_cls = indicator_map[indicator_name]
        hist, bin_edges = pc.compute_histogram(indicator_cls=indicator_cls)
        pc.plot_histogram(hist=hist, bin_edges=bin_edges, indicator_cls=indicator_cls)

        hist_ref, bin_edges_ref = pc_ref.compute_histogram(indicator_cls=indicator_cls)
        pc_ref.plot_histogram(
            hist=hist_ref, bin_edges=bin_edges_ref, indicator_cls=indicator_cls
        )
        print(f"  L1         : {hist_dist.l1(hist, hist_ref):.4f}")
        print(f"  L2         : {hist_dist.l2(hist, hist_ref):.4f}")
        print(f"  Chi²       : {hist_dist.chi_square(hist, hist_ref):.4f}")
        print(f"  KL         : {hist_dist.kl_divergence(hist, hist_ref):.4f}")
        print(f"  Jensen-Shannon : {hist_dist.js_divergence(hist, hist_ref):.4f}")

    # Visualisations et Histogrammes avec normalisation des points
    pc.normalize()  # attention, methode destructrice donc on perd le pc non normalisé
    pc_ref.normalize()  # idem
    for indicator_name in indicators_to_plot:
        indicator_cls = indicator_map[indicator_name]
        hist, bin_edges = pc.compute_histogram(indicator_cls=indicator_cls)
        pc.plot_histogram(hist=hist, bin_edges=bin_edges, indicator_cls=indicator_cls)

        hist_ref, bin_edges_ref = pc_ref.compute_histogram(indicator_cls=indicator_cls)
        pc_ref.plot_histogram(
            hist=hist_ref, bin_edges=bin_edges_ref, indicator_cls=indicator_cls
        )
        print(f"  L1         : {hist_dist.l1(hist, hist_ref):.4f}")
        print(f"  L2         : {hist_dist.l2(hist, hist_ref):.4f}")
        print(f"  Chi²       : {hist_dist.chi_square(hist, hist_ref):.4f}")
        print(f"  KL         : {hist_dist.kl_divergence(hist, hist_ref):.4f}")
        print(f"  Jensen-Shannon : {hist_dist.js_divergence(hist, hist_ref):.4f}")

    # Downsampling et calcul des distances
    pc.downsample()  # attention, destructeur aussi ----> envisager de fair une version non destructrice
    pc_ref.downsample()  # idem
    pc.normalize()  # on renormalise après le downsampling
    pc_ref.normalize()
    distance = Distance(normalize=False)
    for metric in metrics:
        method = getattr(distance, metric)
        try:
            value = method(pc.points, pc_ref.points)
            print(f" {metric} : {value:.4f}")
        except Exception as e:
            print(f" Erreur lors du calcul de {metric} : {e}")
