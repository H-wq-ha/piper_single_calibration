"""Official OpenCV hand-eye calibration and held-out evaluation."""
import csv
import json
import time
import sys
from pathlib import Path

import numpy as np
import cv2
import scipy

from .geometry import inverse, motion_side, pose_error
from .opencv_backend import DEFAULT_METHOD, METHODS as OPENCV_METHODS, fit_opencv, backend_info
from .storage import write_json
from .vision import project


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "std": None, "rmse": None,
                "median": None, "p95": None, "p99": None, "max": None}
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite evaluation values")
    return {"n": len(a), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a)>1 else 0.,
            "rmse": float(np.sqrt(np.mean(a*a))), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def split_samples(samples):
    n = len(samples)
    if n < 10:
        raise ValueError("At least ten samples needed for train/development/test split")
    a, b = max(6, int(n*.6)), int(n*.8)
    b = max(a+1, min(b, n-1))
    return {"train": samples[:a], "development": samples[a:b], "test": samples[b:]}


def predict(G, X, Y, mode):
    return inverse(X) @ motion_side(G, mode) @ Y


def evaluate(samples, mode, K, dist, X, Y):
    pixels, translations, rotations, deltas, components, frame_rows = [], [], [], [], [], []
    for s in samples:
        predicted = predict(s["G"], X, Y, mode)
        t, r = pose_error(predicted, s["C"])
        pixel_delta = project(predicted, s["object_points"], K, dist)-s["image_points"]
        norm = np.linalg.norm(pixel_delta, axis=1)
        delta = (predicted[:3, 3]-s["C"][:3, 3])*1000
        deltas.append(delta)
        components.append(pixel_delta)
        pixels.append(norm)
        translations.append(t)
        rotations.append(r)
        frame_rows.append({"sample_id": s["id"], "corners": len(norm),
                           "chain_rmse_px": stats(norm)["rmse"], "chain_p95_px": stats(norm)["p95"],
                           "chain_max_px": float(norm.max()), "pose_translation_mm": t,
                           "pose_rotation_deg": r, "delta_camera_x_mm": float(delta[0]),
                           "delta_camera_y_mm": float(delta[1]), "delta_camera_z_mm": float(delta[2]),
                           "visual_fit_rmse_px": s["quality"]["fit_rmse_px"]})
    all_pixels = np.concatenate(pixels) if pixels else []
    result = {"frames": len(samples), "chain_pixel": stats(all_pixels),
              "pose_translation_mm": stats(translations), "pose_rotation_deg": stats(rotations),
              "frame_rmse_px": stats([r["chain_rmse_px"] for r in frame_rows]),
              "fraction_corners_above_1px": float(np.mean(np.asarray(all_pixels)>1)) if len(all_pixels) else None,
              "fraction_corners_above_2px": float(np.mean(np.asarray(all_pixels)>2)) if len(all_pixels) else None}
    if deltas:
        delta_array = np.asarray(deltas)
        pixel_array = np.concatenate(components)
        result["signed_pose_delta_camera_mm"] = {axis: stats(delta_array[:, k]) for k, axis in enumerate("xyz")}
        result["signed_pixel_delta"] = {axis: stats(pixel_array[:, k]) for k, axis in enumerate(["u", "v"])}
    # Frame-cluster bootstrap; descriptive, conditional on this fitted model.
    if len(pixels) > 1:
        rng = np.random.default_rng(20261018)
        agg = np.array([[len(g), np.sum(g), np.sum(g*g)] for g in pixels])
        total = agg[rng.integers(0, len(pixels), (2000, len(pixels)))].sum(axis=1)
        result["cluster_ci95"] = {"pixel_mean": np.percentile(total[:, 1]/total[:, 0], [2.5, 97.5]).tolist(),
            "pixel_rmse": np.percentile(np.sqrt(total[:, 2]/total[:, 0]), [2.5, 97.5]).tolist(),
            "note": "Frame bootstrap, fixed fitted model; does not capture fitting uncertainty or session correlation."}
    return result, frame_rows


def fit_candidates(train, development, mode, K, dist, method=DEFAULT_METHOD):
    """Execute official OpenCV solvers and select the requested method."""
    models, failures = {}, {}
    for name in OPENCV_METHODS:
        start = time.perf_counter()
        try:
            X, Y, diag = fit_opencv(train, mode, name)
            for sample in train:
                T = predict(sample["G"], X, Y, mode)
                obj = sample.get("object_points")
                if obj is not None and np.min((obj @ T[:3, :3].T+T[:3, 3])[:, 2]) <= 0:
                    raise ValueError("Solver produced a board behind the training camera")
            models[name] = {"X": X, "Y": Y, "diagnostics": diag,
                "seconds": time.perf_counter()-start, "training_ids": [s["id"] for s in train],
                "backend": diag["backend"],
                "custom_post_refinement": False}
        except (ValueError, RuntimeError, np.linalg.LinAlgError, cv2.error) as error:
            failures[name] = str(error)
    official = {name: model for name, model in models.items() if name in OPENCV_METHODS}
    if not official:
        raise RuntimeError(f"All official OpenCV solvers failed: {failures}")
    scores = {name: evaluate(development, mode, K, dist, m["X"], m["Y"])[0]["chain_pixel"]["rmse"]
              for name, m in models.items()}
    if method == "auto":
        selected = min(official, key=lambda name: scores[name])
    elif method in official:
        selected = method
    elif method not in OPENCV_METHODS:
        raise ValueError("Unknown official solver: "+method)
    else:
        raise RuntimeError(f"Requested official solver {method} failed: {failures.get(method)}; no silent fallback")
    return models, failures, selected, scores


def run_evaluation(meta, samples, output, method=DEFAULT_METHOD):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups = {role: [s for s in samples if s.get("role") == role] for role in ["train", "development", "test"]}
    if any("role" in s for s in samples) and not all(groups.values()):
        raise ValueError("Recorded roles need nonempty train/development/test groups; explicitly auto-split if intended")
    if not all(groups.values()):
        groups = split_samples(samples)
    K, dist, mode = np.asarray(meta["K"], float), np.asarray(meta["dist"], float), meta["mode"]
    models, failures, selected, scores = fit_candidates(groups["train"], groups["development"], mode, K, dist, method=method)
    result = {"dataset": meta.get("dataset", "recorded_session"), "mode": mode, "K": K, "dist": dist,
              "runtime": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                          "numpy": np.__version__, "scipy": scipy.__version__},
              "metadata": meta, "split": {k: [s["id"] for s in v] for k, v in groups.items()},
              "selection_metric": ("development chain pixel RMSE among official methods only" if method == "auto" else "explicit fixed official method; development scores are descriptive"),
              "requested_method": method, "backend": backend_info(),
              "selected_method": selected, "development_scores": scores, "failures": failures, "methods": {},
              "limits": ["Pose consistency does not establish independent TCP/hand-eye accuracy.",
                         "All held-out detected corners are reported, including PnP-rejected corners.",
                         "Contiguous 60/20/20 split is one split of one acquisition session, not cross-session industrial certification."]}
    rows = []
    for name, model in models.items():
        report = {**model, "metrics": {}}
        if meta.get("X_ground_truth") is not None:
            t, r = pose_error(model["X"], np.asarray(meta["X_ground_truth"]))
            report["ground_truth_error"] = {"translation_mm": t, "rotation_deg": r}
        for role, subset in groups.items():
            metrics, frames = evaluate(subset, mode, K, dist, model["X"], model["Y"])
            report["metrics"][role] = metrics
            rows.extend({"method": name, "split": role, **f} for f in frames)
        result["methods"][name] = report
    write_json(output/"evaluation.json", result)
    with (output/"per_frame.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    export_calibration(output, mode, models[selected], selected)
    return result


def export_calibration(output, mode, best, selected):
    """Persist the selected official result for a recorded session."""
    output = Path(output)
    solver_backend = best.get("backend", best.get("diagnostics", {}).get("backend", backend_info()))
    write_json(output/"calibration.json", {"schema_version": 1, "mode": mode,
        "X_name": "T_cam2base" if mode == "eye_to_hand" else "T_cam2gripper",
        "X": best["X"], "Y": best["Y"], "selected_method": selected,
        "validated": "held-out consistency only; no physical acceptance", "translation_unit": "m", "backend": solver_backend})
    import yaml
    suffix = "cam2base" if mode == "eye_to_hand" else "cam2gripper"
    payload = {"R_"+suffix: best["X"][:3, :3].tolist(), "t_"+suffix: best["X"][:3, 3].tolist(),
               "T_"+suffix: best["X"].tolist(), "T_target2gripper" if mode == "eye_to_hand" else "T_target2base": best["Y"].tolist(),
               "mode": mode, "method": selected, "translation_unit": "m",
               "physical_acceptance": False, "backend": solver_backend}
    (output/"handeye.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
