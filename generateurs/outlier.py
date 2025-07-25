import numpy as np


def generate_outliers_batch_v2(
    corolla_batch,
    n_outliers=1000,
    lateral_std=0.05,  # dispersion latérale XZ
    vertical_std=0.02,  # dispersion verticale Y
    y_jitter_centered=True,  # True = bruit centré sur le point, False = uniquement vers le bas/haut
):
    """
    Génère des outliers "réalistes" en ajoutant un petit bruit dans XZ et Y
    autour de points de la corolle, comme dans les données réelles captées.

    Args:
        corolla_batch (np.ndarray): shape (n, N, 3), batch de corolles.
        n_outliers (int): nombre d’outliers par corolle.
        lateral_std (float): écart-type du bruit dans les directions X et Z.
        vertical_std (float): écart-type du bruit en Y (vertical).
        y_jitter_centered (bool): si False, décalage vertical toujours vers le bas (comme des ombres).

    Returns:
        np.ndarray: (n, n_outliers, 3) outliers générés.
    """
    corolla_batch = np.asarray(corolla_batch)
    if corolla_batch.ndim == 2:
        corolla_batch = corolla_batch[np.newaxis, :, :]  # (1, N, 3)

    n, N, _ = corolla_batch.shape

    # Tirage aléatoire de points de base sur chaque corolle
    base_indices = np.random.randint(0, N, size=(n, n_outliers))
    base_points = corolla_batch[
        np.arange(n)[:, None], base_indices
    ]  # shape (n, n_outliers, 3)

    # Bruit latéral (XZ)
    offset_xz = np.random.normal(scale=lateral_std, size=(n, n_outliers, 2))

    # Bruit vertical
    if y_jitter_centered:
        offset_y = np.random.normal(scale=vertical_std, size=(n, n_outliers, 1))
    else:
        offset_y = -np.abs(
            np.random.normal(scale=vertical_std, size=(n, n_outliers, 1))
        )  # toujours vers le bas

    # Construction des offsets
    offsets = np.zeros_like(base_points)
    offsets[..., 0] = offset_xz[..., 0]  # X
    offsets[..., 1] = offset_y[..., 0]  # Y
    offsets[..., 2] = offset_xz[..., 1]  # Z

    outliers = base_points + offsets

    return outliers.squeeze(0) if outliers.shape[0] == 1 else outliers


def generate_outliers_batch_v3(
    corolla_batch,
    n_outliers=1000,
    lateral_std=0.05,
    vertical_std=0.02,
    y_jitter_centered=True,
    min_outliers=10,
):
    corolla_batch = np.asarray(corolla_batch)
    if corolla_batch.ndim == 2:
        corolla_batch = corolla_batch[np.newaxis, :, :]  # shape (1, N, 3)

    n, N, _ = corolla_batch.shape
    valid_mask = corolla_batch[:, :, 1] >= 0.45
    valid_counts = valid_mask.sum(axis=1)

    k = n_outliers
    while np.any(valid_counts < k):
        k //= 2
        if k < min_outliers:
            raise ValueError(
                f"Impossible de générer au moins {min_outliers} outliers avec y ≥ 0.45 pour toutes les corolles."
            )

    rand = np.random.rand(n, N)
    rand[~valid_mask] = np.inf
    topk_indices = np.argpartition(rand, k, axis=1)[:, :k]
    base_points = np.take_along_axis(corolla_batch, topk_indices[:, :, None], axis=1)

    offset_xz = np.random.normal(scale=lateral_std, size=(n, k, 2))
    offset_y = (
        np.random.normal(scale=vertical_std, size=(n, k, 1))
        if y_jitter_centered
        else -np.abs(np.random.normal(scale=vertical_std, size=(n, k, 1)))
    )

    offsets = np.zeros_like(base_points)
    offsets[..., 0] = offset_xz[..., 0]
    offsets[..., 1] = offset_y[..., 0]
    offsets[..., 2] = offset_xz[..., 1]

    outliers = base_points + offsets
    return outliers.squeeze(0) if outliers.shape[0] == 1 else outliers
