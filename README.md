# 3D Point Cloud Generation and Comparison

Python tools for generating synthetic 3D point clouds of vegetated **Urban Canopée corollas** and comparing them with reconstructed, real-world point clouds.

The project was developed to support a 3D plant segmentation workflow: create labelled synthetic examples, measure how closely they resemble real acquisitions, and explore the effects of geometric noise and data augmentation.

## Features

- **Synthetic data generation:** sample points from a corolla mesh, simulate vegetation with either a simple (random-sphere) or iterative growth generator, and optionally add outliers.
- **Point-cloud I/O and processing:** work with labelled `.npy` clouds and `.ply` reconstructions; normalize, transform, and visualize them.
- **Geometric indicators:** analyze nearest-neighbor distances, distances to the vertical (Y) axis, and the distribution of Y coordinates.
- **Cloud comparison:** evaluate differences using Chamfer distance, Earth Mover's Distance / optimal transport, and Sinkhorn-based distances.
- **Histogram comparison:** compare indicator distributions using L1, L2, chi-squared, KL, Jensen–Shannon, and Earth Mover distances.
- **Batch evaluation:** compare collections of synthetic clouds to real reference clouds and summarize distances, including their mean.

## Project structure

The main modules are organized as follows (some paths may differ between versions of the repository):

```text
.
├── main.py                    # Example analysis / entry point
├── point_cloud.py             # Point-cloud representations and loading
├── data_augmentation.py      # Geometric perturbations and augmentation
├── generateurs/
│   ├── generateurs_naif.py    # Naive synthetic vegetation generator
│   ├── generateurs_iteratif.py # Iterative plant-growth generator
│   └── utils_gen.py           # Generation utilities
├── metrics/
│   ├── metrics.py             # Point-cloud distance computations
│   ├── indicators.py          # Local and geometric indicators
│   └── histogram_distance.py  # Histogram distance functions
├── compare/                   # Comparisons against reference clouds
└── requirements.txt           # Python dependencies
```

## Installation

Use Python and a virtual environment. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

The core workflow uses scientific Python tools such as **NumPy**, **SciPy**, **Matplotlib**, **PyVista**, **Open3D**, and **POT** (`ot`). Some generation or mesh-processing steps also use **trimesh** and **pyransame**. Refer to `requirements.txt` for the dependencies required by your checkout.

## Usage

### 1. Prepare point clouds

- **Synthetic data:** generate examples from the corolla mesh using either generator in `generateurs/`. Generated clouds can include corolla points, vegetation points, and artificial outliers.
- **Real data:** provide reconstructed `.ply` clouds from real installations.

The synthetic `.npy` datasets use the following class convention:

| Label | Class |
| --- | --- |
| `0` | Corolla structure |
| `1` | Vegetation |
| `2` | Outlier |

When comparing clouds, use consistent coordinate systems, orientations, and normalization. The existing processing workflow includes centering/scaling clouds into a common frame; some real acquisitions also require an X-axis rotation for alignment.

### 2. Configure the analysis

Open `main.py` and set the input paths, selected point clouds, indicators, and distance methods for the experiment you want to run. The project currently uses a Python-script workflow rather than a documented command-line interface.

### 3. Run

```bash
python3 main.py
```

Depending on the script configuration, the analysis can display point clouds, plot indicator histograms, or compute distances between individual clouds and sets of references. Save generated results and figures in the output directories configured by the scripts.

## Methods

### Synthetic generation

The corolla is sampled from a 3D mesh. Two vegetation-generation approaches are available:

1. **Naive generator:** distributes points in randomly positioned spherical regions, providing a simple synthetic baseline.
2. **Iterative generator:** grows vegetation from initial seeds in successive steps, with configurable density, radius, and stopping conditions.

Outlier injection and geometric transformations can be used to make the synthetic observations less idealized.

### Indicators and distances

| Level | Methods | Purpose |
| --- | --- | --- |
| Point-wise geometry | Nearest-neighbor distance, distance to Y axis, Y-coordinate distribution | Describe local density and spatial organization |
| Cloud-to-cloud | Chamfer, Earth Mover / optimal transport, Sinkhorn | Compare 3D geometric distributions |
| Histogram-to-histogram | L1, L2, chi-squared, KL, Jensen–Shannon, EMD | Compare statistical distributions of indicators |

These metrics have different assumptions and computational costs; results should be interpreted together rather than treating one distance as a universal quality score.

## Research context

This repository focuses on **synthetic data generation and quantitative evaluation**. It supports a broader experiment on separating vegetation, corolla structure, and outliers in reconstructed 3D point clouds. Segmentation-model training (such as PointMLP) can use the generated labelled data, but is not required to run the comparison tools.

## Notes

- Large point clouds and optimal-transport calculations can be memory- and time-intensive; subsampling may be necessary.
- `.ply` and `.npy` sources can differ in coordinate orientation, scale, and label availability.
- Dataset paths and output paths may need to be edited locally before running `main.py`.

