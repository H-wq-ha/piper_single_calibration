"""ChArUco/checkerboard observations and planar candidate-aware robust PnP."""
import cv2
import numpy as np

from .geometry import transform
from .board import SQUARES, SQUARE_M, MARKER_M, LEGACY_PATTERN, create_board, create_detector


class CharucoFrontend:
    def __init__(self, squares=SQUARES, square_m=SQUARE_M, marker_m=MARKER_M,
                 min_corners=20, legacy_pattern=LEGACY_PATTERN):
        self.board, _ = create_board(squares, square_m, marker_m, legacy_pattern)
        self.detector = create_detector(self.board)
        self.min_corners = min_corners

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if corners is None or ids is None or len(corners) != len(ids) or len(corners) < self.min_corners:
            raise ValueError("Insufficient/mismatched ChArUco corners")
        obj, img = self.board.matchImagePoints(corners, ids)
        return obj.reshape(-1, 3).astype(float), img.reshape(-1, 2).astype(float), ids.flatten()


def checkerboard(image, pattern, square_m):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ok, corners = cv2.findChessboardCornersSB(gray, tuple(pattern),
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        raise ValueError(f"Checkerboard {pattern} not detected")
    img = corners.reshape(-1, 2).astype(float)
    # Explicit image-space origin, consistent for the modest board rotations in these sequences.
    # Symmetric boards cannot identify arbitrary 180-degree physical relabelling.
    if img[0].sum() > img[-1].sum():
        img = img[::-1].copy()
    obj = np.zeros((len(img), 3))
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)*square_m
    return obj, img


def project(T, obj, K, dist):
    return cv2.projectPoints(np.asarray(obj, float), cv2.Rodrigues(T[:3, :3])[0],
                             T[:3, 3], K, dist)[0].reshape(-1, 2)


def estimate_pose(obj, img, K, dist, threshold_px=2., min_fraction=.7):
    obj, img = np.asarray(obj, float), np.asarray(img, float)
    if len(obj) < 6:
        raise ValueError("At least six points required")
    cv2.setRNGSeed(20261018)
    ok, rv, tv, indices = cv2.solvePnPRansac(obj, img, K, dist, iterationsCount=150,
        reprojectionError=threshold_px, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    seed = "epnp_ransac"
    if not ok or indices is None or len(indices) < max(6, int(min_fraction*len(obj))):
        # EPnP RANSAC can be numerically poor on nearly frontoparallel planar boards.
        # Keep the same inlier threshold: use a planar seed, never relax the quality gate.
        alternatives = []
        if np.ptp(obj[:, 2]) < 1e-8:
            result = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
            for r, t in zip(result[1], result[2]):
                T = transform(cv2.Rodrigues(r)[0], t)
                errors = np.linalg.norm(project(T, obj, K, dist)-img, axis=1)
                inside = np.flatnonzero(errors < threshold_px)
                alternatives.append((len(inside), -float(np.median(errors)), r, t, inside))
        if not alternatives:
            raise ValueError("PnP consensus too small")
        count, _, rv, tv, indices = max(alternatives, key=lambda a: (a[0], a[1]))
        if count < max(6, int(min_fraction*len(obj))):
            raise ValueError("Both generic and planar PnP consensus too small")
        seed = "ippe_consensus_fallback"
    keep = indices.flatten()
    candidates = []
    if np.ptp(obj[:, 2]) < 1e-8:
        result = cv2.solvePnPGeneric(obj[keep], img[keep], K, dist, flags=cv2.SOLVEPNP_IPPE)
        for r, t in zip(result[1], result[2]):
            candidates.append((r.copy(), t.copy()))
    candidates.append((rv.copy(), tv.copy()))
    ranked = []
    for r, t in candidates:
        R = cv2.Rodrigues(r)[0]
        if np.min((obj @ R.T+t.reshape(3))[:, 2]) <= 0:
            continue
        initial = transform(R, t)
        initial_rms = float(np.sqrt(np.mean(np.sum((project(initial, obj[keep], K, dist)-img[keep])**2, axis=1))))
        r, t = cv2.solvePnPRefineLM(obj[keep], img[keep], K, dist, r, t)
        T = transform(cv2.Rodrigues(r)[0], t)
        rms = float(np.sqrt(np.mean(np.sum((project(T, obj[keep], K, dist)-img[keep])**2, axis=1))))
        ranked.append((rms, initial_rms, T))
    if not ranked:
        raise ValueError("No positive-depth pose")
    ranked.sort(key=lambda a: a[0])
    rms, _, T = ranked[0]
    distinct = [r for r in ranked[1:] if np.linalg.norm(r[2]-T) > 1e-4]
    alternative_gap = float(distinct[0][0]-rms) if distinct else None
    hull_area = float(cv2.contourArea(cv2.convexHull(img.astype(np.float32))))
    return T, {"corners": len(obj), "inliers": len(keep), "inlier_fraction": len(keep)/len(obj), "seed": seed,
               "fit_rmse_px": rms, "alternative_gap_px": alternative_gap,
               "ambiguous": bool(distinct and alternative_gap < .05), "hull_area_px2": hull_area}, keep
