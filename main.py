from generateurs.generateur_iteratif import gen_dataset
from generateurs.generateur_naif import gen_dataset_naif
from generateurs.generateurs import GenerateurNaif, GenerateurIteratif
from compare.visualisation import visualiser_sample_open3d_robuste
from compare.compare import compare_pc
from metrics.indicators import (
    Indicator,
    NearestNeighborDistanceIndicator,
    DistanceToYAxisIndicator,
    YCoordinateDistributionIndicator,
)
from metrics.histogram_distance import HistogramDistances
from metrics.batch_distance_to_reference import Distance_to_ref
from data_augmentation import DataAugmenter
from metrics.pointcloud_distance import EarthMoverDistance, SinkhornDistance, ChamferDistance
from point_cloud import PointCloud, npypcd, plypcd, dictpcd
from pointclouds import PointClouds, load_from_dicts, load_from_npy, load_from_ply
import copy

if __name__ == "__main__":
    # n = 1
    # dataset = gen_dataset(n, with_outliers=True)
    # compare_pc( pc_path="data/synthetiques/dataset_iter_outlier_dropout/sample_0000.npy", ref_pc_path="data/reel/Chalon_en_Champagne_1_candidates_pointcloud_keep.ply", metrics=["earth_mover", "sinkhorn", "schamfer"], indicators_to_plot=["distance_to_y_axis", "nearest_distance", "y_distribution"])

    # Paramètres de déformation aléatoires pour n corolles (à intégrer dans les paramètres du générateur)
    # sigma = np.random.uniform(0.0005, 0.0015, size=(self.n, 1, 1))
    # trans = np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], size=(self.n, 1, 3))
    # angle = np.random.uniform(0, 2.2, size=(self.n, 1, 1))
    # scale_error = np.random.uniform(-0.05, 0.05, size=(self.n, 1, 1))

    source_corolle="Corolle_L_pour_Canoscan_NonRenf_Filins_MESH.obj"
    n=7
    # save_repository="data/synthetiques/dataset_iteratif_oriente_objet_v2"
    save_repository="data/synthetiques/dataset_naif_oriente_objet_v2"
    pre_gen_augmenter= DataAugmenter(sigma_min=0.0005, sigma_max=0.0015, 
                                            trans=0.05, angle_range_deg=2.2, scale=0.05,
                                            min_dropout_ratio=0, max_dropout_ratio=1, dropout_borne=False,
                                            rd_sigma=True, rd_trans=True, rd_angle=True, rd_scale=True, rd_dropout=False
    )
    post_gen_augmenter=DataAugmenter(sigma_min=0, sigma_max=0, trans=0, angle_range_deg=11, scale=0.25, 
                                       min_dropout_ratio=0, max_dropout_ratio=0.4, dropout_borne=False,
                                       rd_sigma=False, rd_trans=False, rd_angle=True, rd_scale=True, rd_dropout=True)
    with_outliers=True
    # generateur_iteratif=GenerateurIteratif(source_corolle=source_corolle, n=n,n_plants_min=8, n_plants_max=25, save_repository=save_repository,
    #                               pre_gen_augmenter=pre_gen_augmenter, post_gen_augmenter=post_gen_augmenter, 
    #                               with_outliers=with_outliers, n_outliers_min=250, n_outliers_max=1_500,
    #                               min_density=1_500_000,max_density=3_000_000,
    #                               min_radius=0.0215, max_radius=0.0305,
    #                               min_decay_density=0.811, max_decay_density=0.899,
    #                               min_decay_radius=0.965, max_decay_radius=0.989,
    #                               min_depth=75, max_depth=350)

    # generateur_iteratif.gen_dataset()

    generateur_naif=GenerateurNaif(source_corolle=source_corolle, n=n, save_repository=save_repository, pre_gen_augmenter=pre_gen_augmenter, post_gen_augmenter=post_gen_augmenter, with_outliers=with_outliers)

    generateur_naif.gen_dataset()

    for i in range(n):
        pc=load_from_npy(f"data/synthetiques/dataset_naif_oriente_objet_v2/sample_{i:04d}.npy")
        pc.visualiser(filename=f"sample_{i:04d}_test_poo")

    """
    EM_distance_to_ref=Distance_to_ref(synthe_repository="data/synthetiques/dataset_iteratif_oriente_objet", reel_repository="data/reel", metrics=EarthMoverDistance(), max_size_repository=500)
    mean_EM_distances_to_ref=EM_distance_to_ref.mean()
    EM_Y_distibution_distance_to_ref_histo=Indicator_to_ref(synthe_repository="data/synthetiques/dataset_iteratif_oriente_objet", reel_repository="data/reel", metrics=YCoordinateDistributionIndicator(), max_size_repository=500)
    mean_EM_Y_distibution_distances_to_ref_histo=EM_Y_distibution_distance_to_ref_histo.mean()
    """

"""
    source_corolle="Corolle_L_pour_Canoscan_NonRenf_Filins_MESH.obj"
    n=1000
    save_repository="data/synthetiques/dataset_iteratif_oriente_objet"
    pre_gen_augmenter= BatchPCDataAugmenter(sigma_min=0.0005, sigma_max=0.0015, 
                                            trans=0.05, angle_range_deg=2.2, scale=0.05,
                                            min_dropout_ratio=0, max_dropout_ratio=1, dropout_borne=False,
                                            rd_sigma=True, rd_trans=True, rd_angle=True, rd_scale=True, rd_dropout=False
    )
    post_gen_augmenter=PCDataAugmenter(sigma_min=0, sigma_max=0, trans=0, angle_range_deg=11, scale=0.25, 
                                       min_dropout_ratio=0, max_dropout_ratio=0.4, dropout_borne=False,
                                       rd_sigma=False, rd_trans=False, rd_angle=True, rd_scale=True, rd_dropout=True)
    with_outliers=True
    generateur_iteratif=GenerateurIteratif(source_corolle=source_corolle, n=n,n_plants_min=8, n_plants_max=25, save_repository=save_repository,
                                   pre_gen_augmenter=pre_gen_augmenter, post_gen_augmenter=post_gen_augmenter, 
                                   with_outliers=with_outliers, n_outliers_min=250, n_outliers_max=1_500,
                                   min_density=1_500_000,max_density=3_000_000,
                                   min_radius=0.0215, max_radius=0.0305,
                                   min_decay_density=0.811, max_decay_density=0.899,
                                   min_decay_radius=0.965, max_decay_radius=0.989,
                                   min_depth=75, max_depth=350)
   
    generateur_iteratif.gen_dataset()
"""
#    for i in range(n):
#        pc=npypcd(f"data/synthetiques/dataset_iteratif_oriente_objet/sample_{i:04d}.npy")
#        pc.visualiser(save=True, filename=f"sample_{i:04d}_test_poo")

#    central_radius = 0.025
#    central_base_density = 4_000_000
#    central_decay_density = 0.86
#    central_decay_radius = 0.988
#    std_radius = 0.0041
#    min_val = 0.95
#    max_val = 0.99925
#    central_max_depth = 250

#        radius = np.random.uniform(0.8 * central_radius, 1.25 * central_radius)
#        base_density = np.random.uniform(
#            0.85 * central_base_density, 1.25 * central_base_density
#        )
#        decay_density = np.random.uniform(
#            0.85 * central_decay_density, 1.014 * central_decay_density
#        )
#        decay_radius = np.clip(
#            np.random.normal(loc=central_decay_radius, scale=std_radius),
#            min_val,
#            max_val,
#        )

"""
    for i in range(10):
        pc=npypcd(f"data/synthetiques/dataset_naif_oriente_objet/sample_{i:04d}.npy")
        pc.visualiser(save=True, filename=f"sample_{i:04d}_test_poo")
"""
"""
    pc = npypcd("data/synthetiques/dataset_iter_outlier_dropout/sample_0000.npy")
    pc_ref = plypcd("data/reel/Chalon_en_Champagne_1_candidates_pointcloud_keep.ply")

    earthmover=EarthMoverDistance(normalize=True)
    sinkhorn=SinkhornDistance(normalize=True)
    schamfer=SchamferDistance(normalize=True)
    metrics = [earthmover, sinkhorn, schamfer]

    nearest_neighbor=NearestNeighborDistanceIndicator()
    distance_to_y=DistanceToYAxisIndicator()
    y_distribution=YCoordinateDistributionIndicator()
    indicators=[nearest_neighbor, distance_to_y, y_distribution]

    hist_dist = HistogramDistances()

    # Copie de sauvegarde
    pc_raw = copy.deepcopy(pc)
    pc_ref_raw = copy.deepcopy(pc_ref)

    # Visualisations et Histogrammes sans normalisation des points mais avec normalisation des histogrammes
    print("\n--- Visualisation brute ---")
    pc.visualiser(save=True, filename="pc_brut")

    # Paramètres de déformation aléatoires pour n corolles (à intégrer dans les paramètres du générateur)
    # sigma = np.random.uniform(0.0005, 0.0015, size=(self.n, 1, 1))
    # trans = np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05], size=(self.n, 1, 3))
    # angle = np.random.uniform(0, 2.2, size=(self.n, 1, 1))
    # scale_error = np.random.uniform(-0.05, 0.05, size=(self.n, 1, 1))
    rotater=PCDataAugmenter(sigma_min=0.0005, sigma_max=0.0015,
                            trans=0.05, 
                            angle_range_deg=11, 
                            min_dropout_ratio=0,
                            max_dropout_ratio=0.6,
                            scale=0.05)

    rotater.apply_rotation(pc=pc_ref, axis="x", degrees=-90)
    pc_ref.visualiser(save=True, filename="reel_pc_rotated")

    print("\n--- Comparaison des histogrammes (points bruts, hist normalisés) ---")
    for indicator in indicators:
        indicator_name=indicator.__class__.__name__
        print(f"\n Indicateur : {indicator_name}")

        hist, bin_edges = pc.compute_histogram(indicator=indicator)
        pc.plot_histogram(hist=hist, bin_edges=bin_edges, name=pc.name, indicator=indicator)

        hist_ref, bin_edges_ref = pc_ref.compute_histogram(indicator=indicator)
        pc_ref.plot_histogram(
            hist=hist_ref, bin_edges=bin_edges_ref, name=pc_ref.name, indicator=indicator
        )
        print(f"  L1         : {hist_dist.l1(hist, hist_ref):.4f}")
        print(f"  L2         : {hist_dist.l2(hist, hist_ref):.4f}")
        print(f"  Chi²       : {hist_dist.chi_square(hist, hist_ref):.4f}")
        print(f"  KL         : {hist_dist.kl_divergence(hist, hist_ref):.4f}")
        print(f"  Jensen-Shannon : {hist_dist.js_divergence(hist, hist_ref):.4f}")

    # Visualisations et Histogrammes avec normalisation des points
    pc.normalize()  # attention, methode destructrice donc on perd le pc non normalisé
    pc_ref.normalize()  # idem
    pc.visualiser(save=True, filename="pc_normalized")
    pc_ref.visualiser(save=True, filename="ref_pc_normalized")
    print("\n--- Comparaison des histogrammes après normalisation ---")
    for indicator in indicators:
        indicator_name=indicator.__class__.__name__
        print(f"\n Indicateur : {indicator_name}")

        hist, bin_edges = pc.compute_histogram(indicator=indicator)
        pc.plot_histogram(hist=hist, bin_edges=bin_edges, name=pc.name, indicator=indicator)

        hist_ref, bin_edges_ref = pc_ref.compute_histogram(indicator=indicator)
        pc_ref.plot_histogram(
            hist=hist_ref, bin_edges=bin_edges_ref, name=pc_ref.name, indicator=indicator
        )
        print(f"  L1         : {hist_dist.l1(hist, hist_ref):.4f}")
        print(f"  L2         : {hist_dist.l2(hist, hist_ref):.4f}")
        print(f"  Chi²       : {hist_dist.chi_square(hist, hist_ref):.4f}")
        print(f"  KL         : {hist_dist.kl_divergence(hist, hist_ref):.4f}")
        print(f"  Jensen-Shannon : {hist_dist.js_divergence(hist, hist_ref):.4f}")

    print("\n--- Calcul des distances après downsampling + normalisation ---")
    # Downsampling et calcul des distances
    pc_down = copy.deepcopy(pc)
    pc_ref_down = copy.deepcopy(pc_ref)
    downsampler=PCDataAugmenter(sigma_min=0, sigma_max=0, trans=0, angle_range_deg=0, scale=0, min_dropout_ratio=0.9, max_dropout_ratio=0.95, max_points=50_000, rd_dropout=False)
    downsampler.apply_dropout(pc_down)  # attention, destructeur aussi ----> envisager de faire une version non destructrice
    downsampler.apply_dropout(pc_ref_down) #idem
    pc_down.normalize()  # on renormalise après le downsampling
    pc_ref_down.normalize()
    print(f"number of points after downsampling {pc_down.size} for {pc_down.name}")
    print(f"number of points after downsampling {pc_ref_down.size} for {pc_ref_down.name}")
    pc_down.visualiser(save=True, filename="pc_downsampled_renormalized")
    pc_ref_down.visualiser(save=True, filename="pc_reel_downsampled_renormalized")
    for metric in metrics:
        metric_name=metric.__class__.__name__
        try:
            value = metric.compute(pc_down, pc_ref_down)
            print(f"\n--- Distance : {metric_name} ---")
            print(f" Valeur : {value:.4f}")
        except Exception as e:
            print(f" Erreur lors du calcul de {metric_name} : {e}")
"""