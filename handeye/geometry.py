"""All transforms use column vectors and translations in metres."""
import numpy as np
from scipy.spatial.transform import Rotation


def transform(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t).reshape(3)
    return T


def inverse(T):
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def validate_transform(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Expected finite 4x4 transform")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous row")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=2e-5):
        raise ValueError("Rotation is not orthogonal")
    if not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=2e-5):
        raise ValueError("Rotation determinant is not +1")
    return T


def pack(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def unpack(p):
    return transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6])


def mean_transform(Ts):
    return transform(Rotation.from_matrix(np.array([t[:3, :3] for t in Ts])).mean().as_matrix(),
                     np.mean([t[:3, 3] for t in Ts], axis=0))


def pose_error(predicted, measured):
    return (float(np.linalg.norm(predicted[:3, 3]-measured[:3, 3])*1000),
            float(np.rad2deg(Rotation.from_matrix(predicted[:3, :3].T @ measured[:3, :3]).magnitude())))


def motion_side(G, mode):
    if mode == "eye_to_hand":
        return G
    if mode == "eye_in_hand":
        return inverse(G)
    raise ValueError("Mode must be eye_to_hand or eye_in_hand")
