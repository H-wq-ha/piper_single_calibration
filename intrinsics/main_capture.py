"""ChArUco intrinsics calibration with automatic, quality-gated image capture."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import yaml

# Keep direct invocation from any working directory supported.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from handeye.board import board_spec, create_board, create_detector
from intrinsics.autocapture import AutoCaptureGate, frame_quality, view_descriptor

IMAGE_ROOT = ROOT / "data/images"
SAVE_PATH = ROOT / "data/result/calibration.yaml"
MIN_CORNERS = 20
WINDOW = "ChArUco Auto Capture"


@contextmanager
def image_directory_lock():
    """Do not clear images while another capture or calibration is using them."""
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with (IMAGE_ROOT / ".capture.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("另一个内参程序正在运行，请先结束该程序后再启动。") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def clear_previous_images(root, keep_directory):
    """Remove old PNG captures, keeping the current batch and non-image files."""
    root = Path(root).resolve()
    keep_directory = Path(keep_directory).resolve()
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        # Do not traverse directory symlinks into any outside image collection.
        if not path.parent.resolve().is_relative_to(root):
            continue
        if path.parent.resolve().is_relative_to(keep_directory):
            continue
        path.unlink()
        removed += 1
    return removed


def create_charuco_board():
    return create_board()


def detect_charuco(detector, gray):
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(corners) != len(ids) or not len(corners):
        return None, None
    return corners, ids


def save_frame(image, directory, index, corners, ids, metrics, automatic):
    """Commit the untouched camera frame and its observations before counting it."""
    name = f"{index:04d}.png"
    path = directory / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")
    record = {"image": name, "captured_at": datetime.now().astimezone().isoformat(),
              "automatic": automatic, "quality": metrics,
              "corners": np.asarray(corners).reshape(-1, 2).tolist(),
              "ids": np.asarray(ids).flatten().tolist()}
    try:
        with (directory / "samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    except Exception:
        path.unlink()
        raise
    print(f"[SAVE {'AUTO' if automatic else 'MANUAL'}] {path} "
          f"corners={metrics['corners']} sharpness={metrics['sharpness']:.1f}", flush=True)
    return path


def capture_frames_from_realsense(board, directory, args):
    import pyrealsense2 as rs

    detector = create_detector(board)
    gate = AutoCaptureGate(args.stable_seconds, args.interval,
                           args.motion_px, args.view_change)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    started = False
    saved = []
    automatic = not args.manual
    notice = ""
    notice_until = 0.
    try:
        profile = pipeline.start(config)
        started = True
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        session = {"created_at": datetime.now().astimezone().isoformat(),
                   "board": board_spec(), "image_size": [640, 480], "fps": 30,
                   "opencv": cv2.__version__,
                   "stream_intrinsics": {k: getattr(intrinsics, k) for k in
                                         ["width", "height", "fx", "fy", "ppx", "ppy", "coeffs"]},
                   "capture_settings": {k: getattr(args, k) for k in
                                        ["stable_seconds", "interval", "motion_px", "view_change",
                                         "min_sharpness", "min_area", "min_corners", "manual"]}}
        (directory / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        removed = clear_previous_images(IMAGE_ROOT, directory)
        session["previous_images_removed"] = removed
        (directory / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
        print(f"本次启动已清理 {removed} 张旧标定照片。", flush=True)
        print(f"照片目录: {directory}", flush=True)
        print("自动保存已开启。" if automatic else "手动保存模式。", flush=True)
        print("缓慢移动标定板到不同位置、距离和倾角，停稳后自动保存。", flush=True)
        print("A: 暂停/恢复自动保存；SPACE: 手动保存合格帧；Q/关闭窗口: 结束并标定。", flush=True)
        # Let exposure settle before evaluating the first view.
        warmup_until = time.monotonic() + 2.
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            frame = frames.get_color_frame()
            if not frame:
                continue
            raw = np.asanyarray(frame.get_data()).copy()
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            corners, ids = detect_charuco(detector, gray)
            valid, status, metrics = frame_quality(
                gray, corners, ids, args.min_corners, args.min_sharpness, args.min_area)
            now = time.monotonic()
            descriptor = None
            ready = False
            if valid:
                descriptor = view_descriptor(board, corners, ids, (raw.shape[1], raw.shape[0]))
                ready, status = gate.consider(corners, ids, descriptor, now)
            else:
                gate.reset_motion()
            if now < warmup_until:
                ready = False
                gate.reset_motion()
                status = "Camera warming up"
            display = raw.copy()
            if corners is not None:
                for x, y in corners.reshape(-1, 2):
                    cv2.circle(display, (round(float(x)), round(float(y))), 2, (255, 0, 0), -1)
            lines = [f"{'AUTO' if automatic else 'PAUSED'} | Saved: {len(saved)} | Corners: {metrics['corners']}",
                     notice if now < notice_until else status,
                     f"Sharpness: {metrics.get('sharpness', 0):.0f} | Area: {100*metrics.get('area_fraction', 0):.1f}%",
                     "A: auto on/off | SPACE: save | Q: calibrate"]
            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, 27+27*i), cv2.FONT_HERSHEY_SIMPLEX,
                            .52, (0, 220, 0) if valid else (0, 140, 255), 2)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("a"), ord("A")):
                automatic = not automatic
                gate.reset_motion()
                ready = False
            manual_save = key == ord(" ")
            if (automatic and ready) or (manual_save and valid and descriptor is not None):
                saved.append(save_frame(raw, directory, len(saved), corners, ids, metrics,
                                        automatic=not manual_save))
                gate.mark_saved(descriptor, now)
                notice = f"SAVED #{len(saved)} - move / tilt to a new view"
                notice_until = now + 1.
            elif manual_save:
                print(f"[SKIP] {status}", flush=True)
    except KeyboardInterrupt:
        print("\n采集停止，保留已保存照片。", flush=True)
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
    return saved


def calibrate_camera(board, image_paths, image_size=None):
    detector = create_detector(board)
    object_points, image_points, used = [], [], []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"无法读取照片: {path}")
        size = (img.shape[1], img.shape[0])
        if image_size is None:
            image_size = size
        if size != tuple(image_size):
            raise ValueError(f"照片分辨率不一致: {path}: {size} != {image_size}")
        corners, ids = detect_charuco(detector, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if corners is None or len(corners) < MIN_CORNERS:
            print(f"{path} SKIP (角点不足)", flush=True)
            continue
        obj, img_pts = board.matchImagePoints(corners, ids)
        object_points.append(obj)
        image_points.append(img_pts)
        used.append(str(Path(path).resolve()))
        print(f"{path} OK ({len(corners)} corners)", flush=True)
    if len(used) < 5:
        raise RuntimeError(f"有效照片仅 {len(used)} 张，至少需要 5 张；建议采集 20–40 个不同视角。照片已保留。")
    ret, K, dist, rvecs, tvecs, std_intrinsics, _, per_view = cv2.calibrateCameraExtended(
        object_points, image_points, image_size, None, None)
    if not (np.isfinite(ret) and np.isfinite(K).all() and np.isfinite(dist).all()
            and K[0, 0] > 0 and K[1, 1] > 0):
        raise RuntimeError("标定结果无效；照片已保留，原有内参未覆盖。")
    print("Camera Matrix:\n", K, "\nDistortion:\n", dist, flush=True)
    print(f"Reprojection RMS: {ret:.6f} px", flush=True)
    return {"K": K, "dist": dist, "error": float(ret), "image_size": list(image_size),
            "image_paths": used, "std_intrinsics": std_intrinsics.flatten().tolist(),
            "per_view_rms_px": per_view.flatten().tolist()}


def save_result(result, path):
    path = Path(path)
    data = {"camera_matrix": result["K"].tolist(), "dist_coeff": result["dist"].tolist(),
            "reprojection_error": result["error"], "board": board_spec(),
            "image_size": result["image_size"], "image_count": len(result["image_paths"]),
            "calibrated_at": datetime.now().astimezone().isoformat(), "opencv": cv2.__version__,
            "source_images": result["image_paths"],
            "std_deviations_intrinsics": result["std_intrinsics"],
            "per_view_rms_px": result["per_view_rms_px"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(f"{path.stem}.backup_{datetime.now():%Y%m%d_%H%M%S_%f}{path.suffix}")
        shutil.copy2(path, backup)
        print(f"原内参备份: {backup}", flush=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description="ChArUco 自动采集和内参标定 (checker 18 mm, marker 13 mm)")
    parser.add_argument("--calibrate-only", action="store_true", help="仅标定已保存照片，默认使用最新非空批次")
    parser.add_argument("--images-dir", type=Path, help="采集时指定新的照片目录；仅标定时指定已有目录")
    parser.add_argument("--output", type=Path, default=SAVE_PATH, help="内参输出，覆盖前自动备份")
    parser.add_argument("--manual", action="store_true", help="使用空格手动保存，默认自动保存")
    parser.add_argument("--stable-seconds", type=float, default=.6, help="自动保存前静止时长")
    parser.add_argument("--interval", type=float, default=1.5, help="自动保存最小间隔（秒）")
    parser.add_argument("--motion-px", type=float, default=1.5, help="静止角点位移阈值（像素）")
    parser.add_argument("--view-change", type=float, default=.04, help="新视角投影变化阈值（图像归一化距离）")
    parser.add_argument("--min-sharpness", type=float, default=50., help="板内清晰度阈值，可按现场调节")
    parser.add_argument("--min-area", type=float, default=.04, help="角点凸包占图像的最小比例")
    parser.add_argument("--min-corners", type=int, default=MIN_CORNERS)
    args = parser.parse_args()
    if (args.stable_seconds < 0 or args.interval < 0 or args.motion_px <= 0
            or args.view_change <= 0 or args.min_sharpness < 0
            or not 0 < args.min_area < 1 or not MIN_CORNERS <= args.min_corners <= 70
            or not all(np.isfinite(getattr(args, k)) for k in
                       ["stable_seconds", "interval", "motion_px", "view_change", "min_sharpness", "min_area"])):
        parser.error("时间/清晰度必须非负，位移/视角阈值必须为正，面积比例在 (0,1)，角点数在 20–70。")
    return args


def latest_images_directory():
    batches = sorted(p for p in IMAGE_ROOT.glob("*") if p.is_dir() and any(p.glob("*.png")))
    if batches:
        return batches[-1]
    return IMAGE_ROOT


def run(args):
    board, _ = create_charuco_board()
    if args.calibrate_only:
        directory = args.images_dir or latest_images_directory()
        images = sorted(directory.glob("*.png"))
    else:
        directory = args.images_dir or IMAGE_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory.mkdir(parents=True, exist_ok=False)
        images = capture_frames_from_realsense(board, directory, args)
    if not images:
        print(f"尚无已保存照片: {directory}；未修改现有内参。", flush=True)
        return
    print(f"共 {len(images)} 张照片，开始标定...", flush=True)
    result = calibrate_camera(board, images)
    save_result(result, args.output)
    print(f"标定完成: {args.output}\n原始照片: {directory}", flush=True)


def main():
    args = parse_args()
    with image_directory_lock():
        run(args)


if __name__ == "__main__":
    main()
