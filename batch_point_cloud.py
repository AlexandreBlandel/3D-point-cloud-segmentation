import numpy as np
from point_cloud import PointCloud, dictpcd

class BatchPointCloud:
    def __init__(self, points: np.ndarray, colors: np.ndarray, labels: np.ndarray = None):
        assert points.shape == colors.shape
        self.points = points  # shape (n, N, 3)
        self.colors = colors  # shape (n, N, 3)
        self.labels = labels  # shape (n, N) or None

    @classmethod
    def from_pointcloud(cls, pc: PointCloud, n: int):
        """
        Copy a point cloud in a batch of size n
        """
        points = np.tile(pc.points[None, :, :], (n, 1, 1))
        colors = np.tile(pc.colors[None, :, :], (n, 1, 1))
        try:
            labels = np.tile(pc.get_labels()[None, :], (n, 1))
        except:
            labels = None
        return cls(points, colors, labels)
    
    def __getitem__(self, idx:int):
        """
        Get the i-th PointCloud of the batch
        """
        points=self.points[idx]
        colors=self.colors[idx]
        if self.labels is not None:
            labels=self.labels[idx]
        else:
            labels=None
        pc_dict={"points": points, "labels": labels}
        pc=dictpcd(pc_dict)
        return(pc)


#Idea: normalize?
#Idea: max_size for batch (500?) so that it doesn't kill because too mush space was needed -> create several batches if n>max_size