"""RealSense/Piper acquisition. Imported only by the explicit capture command.

No robot movement commands are issued. Enabling the arm requires --enable-arm.
"""
from collections import deque
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .geometry import pose_error, transform
from .storage import SessionWriter, digest, serializable
from .vision import CharucoFrontend, estimate_pose
from .board import board_spec


class PiperReader:
    def __init__(self, port, enable_arm=False):
        from piper_sdk import C_PiperInterface_V2
        self.robot = C_PiperInterface_V2(port)
        self.robot.ConnectPort()
        self.last_stamp = None
        self.last_change = 0.
        if enable_arm:
            end = time.monotonic()+10
            while not self.robot.EnablePiper():
                if time.monotonic() > end:
                    raise TimeoutError("Arm enable timed out")
                time.sleep(.05)

    def snapshot(self):
        message = self.robot.GetArmEndPoseMsgs()
        pose = message.end_pose
        raw = {k: int(getattr(pose, k)) for k in ["X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis"]}
        stamp = float(message.time_stamp)
        hz = float(message.Hz)
        host = time.monotonic()
        if stamp != self.last_stamp:
            self.last_change = host
            self.last_stamp = stamp
        if stamp <= 0 or hz <= 0 or host-self.last_change > .2:
            raise ValueError("No fresh robot feedback")
        t = [raw[k]*1e-6 for k in ["X_axis", "Y_axis", "Z_axis"]]
        angles = [raw[k]*1e-3 for k in ["RZ_axis", "RY_axis", "RX_axis"]]
        G = transform(Rotation.from_euler("ZYX", angles, degrees=True).as_matrix(), t)
        return {"G": G, "raw_sdk": raw, "sdk_timestamp_raw": stamp, "sdk_hz": hz,
                "host_monotonic_s": host, "host_wall_time_s": time.time(),
                "feedback_update_age_ms": (host-self.last_change)*1000}

    def close(self):
        disconnect = getattr(self.robot, "DisconnectPort", None)
        if disconnect:
            disconnect()


def capture(args):
    import pyrealsense2 as rs
    source = Path(args.intrinsics)
    calibration = yaml.safe_load(source.read_text())
    K = np.asarray(calibration["camera_matrix"], float)
    dist = np.asarray(calibration["dist_coeff"], float)
    frontend = CharucoFrontend(min_corners=args.min_corners)
    writer = SessionWriter(args.output, {"mode": args.mode, "K": K, "dist": dist,
        "image_size": [640, 480], "fps": 30, "opencv": cv2.__version__,
        "intrinsics_source": str(source.resolve()), "intrinsics_sha256": digest(source),
        "board": board_spec(),
        "pairing": "static acquisition with host-monotonic robot bracket; not exposure-level hardware synchronization",
        "robot_rotation_convention": "Rz(RZ) @ Ry(RY) @ Rx(RX); SDK convention must be verified for hardware",
        "stationary_window_s": args.stationary_window,
        "stationary_translation_mm": args.stationary_mm, "stationary_rotation_deg": args.stationary_deg,
        "max_visual_rmse_px": args.max_rmse_px,
        "hardware_validation": "not commissioned by offline benchmark"})
    print(f"Recording session: {writer.path}", flush=True)
    print("S=train, V=development, T=test, Q=finish. Move robot manually, then wait until stationary.", flush=True)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    robot = None
    history = deque()
    started = False
    try:
        robot = PiperReader(args.can_port, args.enable_arm)
        profile = pipeline.start(config)
        started = True
        # Retain stream intrinsics without recording the camera's device serial number.
        actual_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        from .storage import write_json
        write_json(writer.path/"device.json", {
            "stream_intrinsics": {k: getattr(actual_intrinsics, k) for k in ["width", "height", "fx", "fy", "ppx", "ppy", "coeffs"]}})
        while True:
            before = None
            problem = None
            try:
                before = robot.snapshot()
            except ValueError as e:
                problem = str(e)
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            frame = frames.get_color_frame()
            if not frame:
                continue
            received = time.monotonic()
            image = np.asanyarray(frame.get_data()).copy()
            after = None
            try:
                after = robot.snapshot()
                history.append(after)
                while history and after["host_monotonic_s"]-history[0]["host_monotonic_s"] > args.stationary_window+.2:
                    history.popleft()
            except ValueError as e:
                problem = str(e)
            observation = None
            try:
                obj, img, ids = frontend.detect(image)
                C, quality, keep = estimate_pose(obj, img, K, dist)
                observation = obj, img, ids, C, quality, keep
                if quality["fit_rmse_px"] > args.max_rmse_px:
                    raise ValueError("Visual RMSE exceeds capture limit")
                if quality["ambiguous"]:
                    raise ValueError("Planar pose candidates are ambiguous")
            except ValueError as e:
                problem = str(e)
            stationary = False
            max_translation, max_rotation = None, None
            if after is not None and history:
                errors = [pose_error(after["G"], h["G"]) for h in history]
                max_translation = max(e[0] for e in errors)
                max_rotation = max(e[1] for e in errors)
                span = after["host_monotonic_s"]-history[0]["host_monotonic_s"]
                stationary = span >= args.stationary_window and max_translation <= args.stationary_mm and max_rotation <= args.stationary_deg
            if not stationary and problem is None:
                problem = "Waiting for stationary robot"
            if before is None or after is None or after["host_monotonic_s"]-before["host_monotonic_s"] > .15:
                problem = "Robot/frame bracket unavailable or >150 ms"
            display = image.copy()
            if observation:
                for u, v in observation[1]:
                    cv2.circle(display, (int(u), int(v)), 2, (0, 220, 0), -1)
            cv2.putText(display, (problem or "READY")[:70], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 0, 255) if problem else (0, 200, 0), 1)
            cv2.putText(display, f"Saved {writer.count} | S train V dev T test Q quit", (8, 47), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 200, 0), 1)
            cv2.imshow("Recorded hand-eye capture", display)
            key = cv2.waitKey(1)&255
            if key == ord("q"):
                break
            if key not in [ord("s"), ord("v"), ord("t")]:
                continue
            if problem:
                rejected = writer.path/"rejected"
                rejected.mkdir(exist_ok=True)
                sid = str(time.time_ns())
                if not cv2.imwrite(str(rejected/(sid+".png")), image):
                    raise RuntimeError("Failed to save rejected image")
                with (writer.path/"rejections.jsonl").open("a") as f:
                    f.write(json.dumps({"id": sid, "reason": problem, "host_monotonic_s": received,
                                        "before": before, "after": after}, default=serializable)+"\n")
                print("Rejected and recorded:", problem, flush=True)
                continue
            obj, img, ids, C, quality, keep = observation
            role = {ord("s"): "train", ord("v"): "development", ord("t"): "test"}[key]
            writer.append(image, {"role": role, "G": after["G"], "C": C, "object_points": obj,
                "image_points": img, "corner_ids": ids, "inlier_indices": keep, "quality": quality,
                "robot_before": before, "robot_after": after,
                "frame": {"device_timestamp_ms": frame.get_timestamp(), "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                          "number": frame.get_frame_number(), "host_received_monotonic_s": received},
                "stationary": {"max_translation_mm": max_translation, "max_rotation_deg": max_rotation}})
            print(f"Saved {writer.count}: {role}, corners={len(img)}, RMSE={quality['fit_rmse_px']:.3f}px", flush=True)
    finally:
        if started:
            pipeline.stop()
        if robot is not None:
            robot.close()
        cv2.destroyAllWindows()
        print(f"Session retained: {writer.path}; {writer.count} valid samples", flush=True)
