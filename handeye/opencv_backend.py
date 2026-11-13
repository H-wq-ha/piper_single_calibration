"""Coordinate adapters to the unmodified upstream OpenCV hand-eye solvers.

Canonical model: C_i = inv(X) @ S_i @ Y, with S_i=inv(G_i) for
eye-in-hand and S_i=G_i for eye-to-hand. G_i is always gripper->base.
Only data conversion, diagnostics and output validation live here.
"""
from importlib.metadata import version

import cv2
import numpy as np

from .geometry import inverse, mean_transform, motion_side, transform, validate_transform
from .solvers import diagnostics, pairs

HAND_EYE_METHODS = {
    "opencv_tsai": "CALIB_HAND_EYE_TSAI",
    "opencv_park": "CALIB_HAND_EYE_PARK",
    "opencv_horaud": "CALIB_HAND_EYE_HORAUD",
    "opencv_andreff": "CALIB_HAND_EYE_ANDREFF",
    "opencv_daniilidis": "CALIB_HAND_EYE_DANIILIDIS",
}
WORLD_HAND_EYE_METHODS = {
    "opencv_shah": "CALIB_ROBOT_WORLD_HAND_EYE_SHAH",
    "opencv_li": "CALIB_ROBOT_WORLD_HAND_EYE_LI",
}
METHODS = tuple(HAND_EYE_METHODS)+tuple(WORLD_HAND_EYE_METHODS)
DEFAULT_METHOD = "opencv_shah"


def backend_info():
    if not all(hasattr(cv2, name) for name in ["calibrateHandEye", "calibrateRobotWorldHandEye"]):
        raise RuntimeError("Official hand-eye APIs are missing. Install the project requirements with "
                           "venv/bin/python -m pip install -r requirements.txt; use OpenCV 4.12, not 5.0.")
    return {"library": "OpenCV", "version": cv2.__version__,
            "distribution": "opencv-contrib-python", "distribution_version": version("opencv-contrib-python"),
            "upstream_source": "https://github.com/opencv/opencv/blob/4.12.0/modules/calib3d/src/calibration_handeye.cpp",
            "license": "Apache-2.0", "custom_handeye_solver": False,
            "binary_sha256_note": "Python distribution pinned in requirements.txt; formulas are not reimplemented."}


def rotations_translations(transforms):
    return ([np.ascontiguousarray(T[:3, :3], dtype=np.float64) for T in transforms],
            [np.ascontiguousarray(T[:3, 3:4], dtype=np.float64) for T in transforms])


def fit_opencv(samples, mode, method=DEFAULT_METHOD):
    provenance = backend_info()
    if method not in METHODS:
        raise ValueError("Unknown OpenCV method: "+method)
    if len(samples) < 3:
        raise ValueError("OpenCV needs at least three paired poses")
    for sample in samples:
        validate_transform(sample["G"])
        validate_transform(sample["C"])
    As, _ = pairs(samples, mode, adjacent=False, min_angle_deg=0.)
    diag = diagnostics(As)
    if diag["weak_rotation"] or diag["translation_rank"] < 3:
        raise ValueError("Motion set is unobservable; independent rotation axes are required")
    S = [motion_side(s["G"], mode) for s in samples]
    C = [np.asarray(s["C"], dtype=np.float64) for s in samples]
    rC, tC = rotations_translations(C)
    if method in HAND_EYE_METHODS:
        # OpenCV expects H_i X C_i = Y. H_i = inv(S_i), so output is our X.
        rH, tH = rotations_translations([inverse(T) for T in S])
        flag_name = HAND_EYE_METHODS[method]
        R, t = cv2.calibrateHandEye(rH, tH, rC, tC, method=getattr(cv2, flag_name))
        X = validate_transform(transform(R, t))
        # AX=XB only estimates X; estimate Y by averaging training-frame transforms.
        Y = mean_transform([inverse(T)@X@c for T, c in zip(S, C)])
        api = "cv2.calibrateHandEye"
        convention = "H=inv(S), target2cam=C; native cam2gripper output is canonical X"
    else:
        # Native equation A_i U = V B_i. Set A=C, B=S; U=inv(Y), V=inv(X).
        # Both X and Y are returned by the upstream simultaneous calibration API.
        rS, tS = rotations_translations(S)
        flag_name = WORLD_HAND_EYE_METHODS[method]
        rU, tU, rV, tV = cv2.calibrateRobotWorldHandEye(rC, tC, rS, tS,
                                                      method=getattr(cv2, flag_name))
        U = validate_transform(transform(rU, tU))
        V = validate_transform(transform(rV, tV))
        X, Y = inverse(V), inverse(U)
        api = "cv2.calibrateRobotWorldHandEye"
        convention = "world2cam=C, base2gripper=S; native U=inv(Y), V=inv(X)"
    validate_transform(Y)
    return X, Y, {**diag, "backend": {**provenance, "api": api, "method_constant": flag_name,
        "method_value": int(getattr(cv2, flag_name)), "coordinate_adapter": convention},
        "all_training_frames_used": True, "training_frames": len(samples),
        "custom_post_refinement": False, "frame_consensus": False}
