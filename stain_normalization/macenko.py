"""
Self-implemented Macenko (2009) stain normalization.

Standard optical-density (OD) SVD algorithm: estimate two stain vectors (H and E)
from the extremes of the OD-space point cloud projected onto its top-2 principal
plane, decompose an image into per-pixel stain concentrations under those vectors,
then reconstruct using a *reference* image's stain vectors and 99th-percentile
concentrations so the output matches the reference's color appearance.

No external stain-normalization package needed (avoids the unmaintained StainTools
dependency) - only numpy, which is already available.
"""
import numpy as np

OD_NORM_THRESHOLD = 0.3  # ||OD|| below this is treated as background/white (unstable direction)
ANGULAR_PERCENTILE = 99


def _to_od(img):
    """img: HxWx3 uint8 RGB -> Nx3 float64 optical density."""
    px = img.reshape(-1, 3).astype(np.float64)
    px = np.clip(px, 1, 255)  # avoid log(0)
    return -np.log(px / 255.0)


def _circular_percentile(angles, pct):
    """
    Percentile of a circular (angular) quantity, robust to the cluster straddling the
    +-pi branch cut. Re-centers on the circular mean before taking a linear percentile,
    then shifts back - a naive np.percentile on raw atan2 output breaks whenever the
    true angular cluster wraps around +-pi (confirmed empirically on real H&E tiles here).
    """
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    centered = np.angle(np.exp(1j * (angles - mean_angle)))
    return np.percentile(centered, pct) + mean_angle


def get_stain_matrix(img, od_norm_threshold=OD_NORM_THRESHOLD, angular_percentile=ANGULAR_PERCENTILE):
    """Estimate the 2x3 stain matrix (rows = unit H, E vectors) from one RGB image."""
    OD = _to_od(img)
    mask = np.linalg.norm(OD, axis=1) > od_norm_threshold
    OD_tissue = OD[mask]
    if OD_tissue.shape[0] < 10:
        raise ValueError("Not enough tissue pixels above OD threshold to fit stain matrix")

    # top-2 principal directions of the OD point cloud
    cov = np.cov(OD_tissue, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, [-1, -2]]  # (3, 2), largest eigenvalue first

    proj = OD_tissue.dot(top2)  # (N, 2)
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    min_angle = _circular_percentile(angles, 100 - angular_percentile)
    max_angle = _circular_percentile(angles, angular_percentile)

    v_min = top2.dot(np.array([np.cos(min_angle), np.sin(min_angle)]))
    v_max = top2.dot(np.array([np.cos(max_angle), np.sin(max_angle)]))

    # Hematoxylin vector has the larger red-channel OD component by convention
    if v_min[0] > v_max[0]:
        stain_matrix = np.array([v_min, v_max])
    else:
        stain_matrix = np.array([v_max, v_min])

    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    return stain_matrix  # (2, 3): row 0 = H, row 1 = E


def get_concentrations(img, stain_matrix):
    """
    Per-pixel stain concentrations under a given stain matrix, via the (2x3) matrix's
    pseudo-inverse - a single small matmul, not a per-call iterative lstsq solve (which
    dominated runtime: ~32ms/patch vs ~1ms/patch, since lstsq re-does an SVD-based solve
    over all N~65536 pixels every call instead of reusing that the stain_matrix itself is tiny).
    """
    OD = _to_od(img)
    pinv = np.linalg.pinv(stain_matrix)  # (3, 2)
    return OD.dot(pinv)  # (N, 2)


def fit_reference(ref_images, od_norm_threshold=OD_NORM_THRESHOLD, angular_percentile=ANGULAR_PERCENTILE):
    """
    Fit the target stain matrix + max concentrations from one or more reference images.
    Multiple images are pooled (concatenated OD pixels) before fitting, so the result
    is a single representative reference rather than an average of per-image fits.
    """
    stacked = np.concatenate([np.asarray(im).reshape(-1, 3) for im in ref_images], axis=0)
    # reuse get_stain_matrix on the pooled "image" (shape doesn't matter, it's flattened anyway)
    fake_img = stacked.reshape(-1, 1, 3)
    stain_matrix = get_stain_matrix(fake_img, od_norm_threshold, angular_percentile)
    concentrations = get_concentrations(fake_img, stain_matrix)
    max_c = np.percentile(concentrations, 99, axis=0)
    return stain_matrix, max_c


def normalize(img, ref_stain_matrix, ref_max_c, od_norm_threshold=OD_NORM_THRESHOLD,
              angular_percentile=ANGULAR_PERCENTILE):
    """
    Normalize `img` (HxWx3 uint8 RGB) toward the reference stain appearance.
    Returns an HxWx3 uint8 RGB array.
    """
    h, w = img.shape[:2]
    src_stain_matrix = get_stain_matrix(img, od_norm_threshold, angular_percentile)
    src_c = get_concentrations(img, src_stain_matrix)
    src_max_c = np.percentile(src_c, 99, axis=0)
    src_max_c[src_max_c == 0] = 1e-6

    c_norm = src_c * (ref_max_c / src_max_c)
    od_norm = c_norm.dot(ref_stain_matrix)
    rgb_norm = 255.0 * np.exp(-od_norm)
    rgb_norm = np.clip(rgb_norm, 0, 255).reshape(h, w, 3).astype(np.uint8)
    return rgb_norm


def normalize_fixed_source(img, src_stain_matrix, src_max_c, ref_stain_matrix, ref_max_c):
    """
    Variant of normalize() for generating StainNet training targets: uses a FIXED
    source-side stain matrix + max concentration (pre-fit once from a representative
    source-domain tile pool via fit_reference()) instead of re-estimating them from
    each individual image.

    This matters because StainNet is a pure per-pixel function (1x1 convs, no spatial
    context): if the teacher target for a given source RGB pixel color depends on
    what else happens to be in that particular crop (the standard per-image-adaptive
    Macenko behavior), the same input color maps to inconsistent targets across the
    training set, and L1 regression against a noisy target collapses toward the
    conditional mean - a visibly washed-out, desaturated, low-contrast StainNet output
    despite the training loss looking like it's converging normally. Fixing the
    source-side stats makes the source->teacher mapping a genuinely deterministic
    RGB->RGB function, which a per-pixel network can actually learn.

    Only the *source* side is fixed here - `get_concentrations` below still computes
    genuine per-pixel concentrations for this specific image's content under that
    fixed basis; only the stain matrix and max-concentration scaling are frozen.
    """
    h, w = img.shape[:2]
    src_c = get_concentrations(img, src_stain_matrix)
    c_norm = src_c * (ref_max_c / src_max_c)
    od_norm = c_norm.dot(ref_stain_matrix)
    rgb_norm = 255.0 * np.exp(-od_norm)
    rgb_norm = np.clip(rgb_norm, 0, 255).reshape(h, w, 3).astype(np.uint8)
    return rgb_norm
