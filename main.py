from generateurs.generateur_iteratif import gen_dataset
from generateurs.generateur_naif import gen_dataset_naif
from compare.visualisation import visualiser_sample_open3d_robuste
from compare.compare import compare_pc

if __name__ == "__main__":
    # n = 1
    # dataset = gen_dataset(n, with_outliers=True)
    compare_pc(
        pc_path="data/synthetiques/dataset_iter_outlier_dropout/sample_0000.npy",
        ref_pc_path="data/reel/Chalon_en_Champagne_1_candidates_pointcloud_keep.ply",
        metrics=["earth_mover", "sinkhorn", "schamfer"],
        indicators_to_plot=["distance_to_y_axis", "nearest_distance", "y_distribution"],
    )
