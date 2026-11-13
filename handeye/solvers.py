"""Motion observability diagnostics for the official OpenCV backend."""
from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import inverse, motion_side


def pairs(samples, mode, adjacent=False, min_angle_deg=.5):
    ids = zip(range(len(samples)-1), range(1, len(samples))) if adjacent else combinations(range(len(samples)), 2)
    As, Bs = [], []
    for i, j in ids:
        A = motion_side(samples[i]["G"], mode) @ inverse(motion_side(samples[j]["G"], mode))
        B = samples[i]["C"] @ inverse(samples[j]["C"])
        if not adjacent and np.rad2deg(Rotation.from_matrix(A[:3, :3]).magnitude()) < min_angle_deg:
            continue
        As.append(A)
        Bs.append(B)
    if len(As) < 2:
        raise ValueError("Not enough rotating motion pairs")
    return np.array(As), np.array(Bs)


def diagnostics(As):
    axes = Rotation.from_matrix(As[:, :3, :3]).as_rotvec()
    s = np.linalg.svd(axes, compute_uv=False)
    P = np.concatenate([a[:3, :3]-np.eye(3) for a in As])
    st = np.linalg.svd(P, compute_uv=False)
    return {"motion_pairs": len(As), "rotation_singular_values": s.tolist(),
            "translation_singular_values": st.tolist(), "translation_rank": int(np.linalg.matrix_rank(P)),
            "weak_rotation": bool(len(s)<2 or s[1] < 1e-4*max(s[0], 1e-12))}
