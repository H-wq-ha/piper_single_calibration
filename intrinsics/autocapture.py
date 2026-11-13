"""Quality, stationarity and viewpoint gates for automatic image capture."""
import cv2
import numpy as np


def frame_quality(gray, corners, ids, min_corners=20, min_sharpness=50., min_area=.04):
    if corners is None or ids is None or len(corners) != len(ids) or len(corners) < min_corners:
        return False, "Show more board corners", {"corners": 0 if corners is None else len(corners)}
    points = np.asarray(corners, np.float32).reshape(-1, 2)
    hull = cv2.convexHull(points)
    area = float(cv2.contourArea(hull)/(gray.shape[0]*gray.shape[1]))
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    values = cv2.Laplacian(gray, cv2.CV_64F)[mask != 0]
    sharpness = float(values.var()) if values.size else 0.
    metrics = {"corners": len(points), "area_fraction": area, "sharpness": sharpness}
    if area < min_area:
        return False, "Move board closer", metrics
    if sharpness < min_sharpness:
        return False, "Blurred: hold still / improve light", metrics
    return True, "Quality OK", metrics


def view_descriptor(board, corners, ids, image_size):
    """Project fixed board probes, so partial detections have comparable views."""
    obj, img = board.matchImagePoints(corners, ids)
    normalized = img.reshape(-1, 2)/np.asarray(image_size, float)
    H, _ = cv2.findHomography(obj.reshape(-1, 3)[:, :2], normalized, 0)
    if H is None:
        return None
    width, height = np.asarray(board.getChessboardSize())*board.getSquareLength()
    probes = np.array([[0, 0], [width, 0], [width, height], [0, height],
                       [width/2, height/2]], np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(probes, H).reshape(-1, 2)
    return projected if np.isfinite(projected).all() else None


class AutoCaptureGate:
    def __init__(self, stable_seconds=.6, interval=1.5, motion_px=1.5, view_change=.04):
        self.stable_seconds = stable_seconds
        self.interval = interval
        self.motion_px = motion_px
        self.view_change = view_change
        self.anchor = None
        self.stable_since = None
        self.last_saved = -float("inf")
        self.saved_views = []

    def reset_motion(self):
        self.anchor = None
        self.stable_since = None

    def consider(self, corners, ids, descriptor, now):
        if descriptor is None:
            self.reset_motion()
            return False, "Board geometry unavailable"
        current = {int(i): p.copy() for i, p in zip(np.asarray(ids).flatten(),
                                                   np.asarray(corners).reshape(-1, 2))}
        common = [] if self.anchor is None else sorted(current.keys() & self.anchor.keys())
        stable = len(common) >= max(8, int(.7*len(current)))
        if stable:
            displacement = np.linalg.norm([current[i]-self.anchor[i] for i in common], axis=1)
            stable = (np.median(displacement) <= self.motion_px and
                      np.percentile(displacement, 90) <= 2*self.motion_px)
        if not stable:
            self.anchor = current
            self.stable_since = now
        elapsed = now-self.stable_since
        if elapsed < self.stable_seconds:
            return False, f"Hold still {elapsed:.1f}/{self.stable_seconds:.1f}s"
        if now-self.last_saved < self.interval:
            return False, "Waiting for capture interval"
        if any(float(np.sqrt(np.mean(np.sum((descriptor-old)**2, axis=1)))) < self.view_change
               for old in self.saved_views):
            return False, "Move / tilt board to a NEW view"
        return True, "Ready to save"

    def mark_saved(self, descriptor, now):
        self.last_saved = now
        self.saved_views.append(np.asarray(descriptor).copy())
