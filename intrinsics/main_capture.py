import cv2
import pyrealsense2 as rs
import numpy as np
import os
import sys
import glob
import yaml

# 配置常量
IMAGE_GLOB = "data/images/*.png"
SAVE_PATH = "data/result/calibration.yaml"
DICT_SIZE = (12, 9)  # Charuco Board的尺寸 (列数, 行数)
SQUARE_SIZE = 0.013  # 方格大小（单位：米）
MARKER_SIZE = 0.009  # 标记大小（单位：米）
MIN_CORNERS = 20  # 最小角点数量，用于评估标定帧的有效性

def create_charuco_board():
    """创建Charuco板和字典"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        DICT_SIZE,
        SQUARE_SIZE,
        MARKER_SIZE,
        dictionary
    )
    return board, dictionary

def create_detector(board):
    """创建 Charuco 检测器（OpenCV 4.7+ 新 API，同时完成 marker 检测和角点插值）"""
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    detector_params = cv2.aruco.DetectorParameters()
    return cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

def detect_charuco(detector, gray):
    """检测 Charuco 角点，返回 (corners, ids)

    OpenCV 5.0 返回 (N,2) 形状的角点；防御性截断到两者数量一致。
    """
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(corners) == 0:
        return None, None
    n = min(len(corners), len(ids))
    if n == 0:
        return None, None
    return corners[:n], ids[:n]

def evaluate_frame(corners, ids):
    """判断当前帧是否适合标定"""
    if corners is None or ids is None:
        return False, 0

    count = len(corners)
    if count < MIN_CORNERS:
        return False, count

    return True, count

def draw_feedback(image, valid, count):
    """在画面上叠加提示信息"""
    text = f"Corners: {count}"

    if valid:
        status = "GOOD - Press SPACE to save"
        color = (0, 200, 0)
    else:
        status = "BAD - Move board"
        color = (0, 0, 255)

    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image

def capture_frames_from_realsense(board):
    """从RealSense相机获取图像并进行处理"""
    os.makedirs("data/images", exist_ok=True)
    detector = create_detector(board)

    # RealSense初始化
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("Press SPACE to save, Q to quit")

    index = 0
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        display_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测 Charuco 角点（新 API，内部完成 marker 检测 + 插值）
        corners, ids = detect_charuco(detector, gray)

        # 评估当前帧
        valid, count = evaluate_frame(corners, ids)

        # 可视化检测结果（手工画点：OpenCV 5.0 的 drawDetectedCornersCharuco
        # 对角点 (N,2) 形状断言失败，已弃用）
        if corners is not None:
            for (x, y) in corners.reshape(-1, 2):
                cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)

        image = draw_feedback(image, valid, count)
        cv2.imshow("Charuco Capture", image)

        key = cv2.waitKey(1)

        # 按键控制
        if key == ord(' '):  # 空格键保存图像
            if valid:
                path = os.path.join("data/images", f"{index}.png")
                cv2.imwrite(path, display_image)
                print(f"[SAVE] {path}  corners={count}")
                index += 1
            else:
                print(f"[SKIP] Not enough corners ({count})")

        elif key == ord('q'):  # 按Q退出
            break

    pipeline.stop()
    cv2.destroyAllWindows()

def calibrate_camera(board, image_paths, image_size):
    """从保存的图像执行相机标定并返回结果

    用 matchImagePoints 得到每帧的 (物方点, 像方点) 后交给 cv2.calibrateCamera，
    替代已移除的 cv2.aruco.calibrateCameraCharuco。
    """
    detector = create_detector(board)
    object_points = []  # 每帧 Nx3 物方点（板坐标系，米）
    image_points = []   # 每帧 Nx2 像方点（像素）

    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids = detect_charuco(detector, gray)
        if corners is None or len(corners) < MIN_CORNERS:
            print(f"{path} SKIP")
            continue

        obj, img_pts = board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < MIN_CORNERS:
            print(f"{path} SKIP")
            continue

        object_points.append(obj)
        image_points.append(img_pts)
        print(f"{path} OK")

    if len(object_points) < 5:
        raise RuntimeError("有效帧太少，无法标定")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    print("Camera Matrix:\n", K)
    print("Distortion:\n", dist)
    print("Reprojection Error:", ret)

    return {
        "K": K,
        "dist": dist,
        "error": ret
    }

def save_result(result, path):
    """保存标定结果到文件"""
    data = {
        "camera_matrix": result["K"].tolist(),
        "dist_coeff": result["dist"].tolist(),
        "reprojection_error": float(result["error"])
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)

def main():
    """
    两种模式：
      默认         : 连接 RealSense 采集图像，按 Q 退出后自动执行标定
      --calibrate-only : 跳过采集，直接用 data/images 下已保存的照片标定
    """
    calibrate_only = '--calibrate-only' in sys.argv

    # 创建Charuco板和字典
    board, dictionary = create_charuco_board()

    if not calibrate_only:
        print("\n提示: 按 Q 退出后会自动执行标定；"
              "若只想用已保存的照片重新标定，可加 --calibrate-only 参数\n")
        # 从 RealSense 采集图像
        capture_frames_from_realsense(board)

    # 加载所有收集的图像并进行标定
    images = sorted(glob.glob(IMAGE_GLOB))
    if not images:
        raise RuntimeError("data/images 下没有图像，请先按空格采集"
                           "（或确认在项目根目录下运行）")

    print(f"共 {len(images)} 张照片，开始标定...")
    image_sample = cv2.imread(images[0])
    image_size = (image_sample.shape[1], image_sample.shape[0])
    result = calibrate_camera(board, images, image_size)

    save_result(result, SAVE_PATH)
    print(f"标定完成，结果已保存至 {SAVE_PATH}")

if __name__ == "__main__":
    main()
