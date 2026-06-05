import cv2
import pyrealsense2 as rs
import numpy as np
import os
import glob
import yaml

# 配置常量
IMAGE_GLOB = "data/images/*.png"
SAVE_PATH = "data/result/calibration.yaml"
DICT_SIZE = (12, 9)  # Charuco Board的尺寸 (列数, 行数)
SQUARE_SIZE = 0.017  # 方格大小（单位：米）
MARKER_SIZE = 0.012  # 标记大小（单位：米）
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

        # 检测ArUco marker
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

        if ids is None:
            continue

        # 插值Charuco角点
        retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )

        # 评估当前帧
        valid, count = evaluate_frame(corners, ids)

        # 可视化检测结果
        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(image, corners, ids, (255, 0, 0))

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

def add_corners_to_calibrator(image, corners, ids, all_corners, all_ids):
    """将检测到的角点添加到标定数据中"""
    if corners is None or ids is None:
        return False

    if len(corners) < MIN_CORNERS:
        return False

    all_corners.append(corners)
    all_ids.append(ids)

    return True

def calibrate_camera(all_corners, all_ids, board, image_size):
    """执行相机标定并返回结果"""
    if len(all_corners) < 5:
        raise RuntimeError("有效帧太少，无法标定")

    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        None,
        None
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
    # 创建Charuco板和字典
    board, dictionary = create_charuco_board()

    # 初始化图像收集
    all_corners = []
    all_ids = []
    image_size = None

    capture_frames_from_realsense(board)

    # 加载所有收集的图像并进行标定
    images = glob.glob(IMAGE_GLOB)

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

        # 插值Charuco角点
        retval, corners, ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

        if add_corners_to_calibrator(img, corners, ids, all_corners, all_ids):
            print(f"{path} OK")
        else:
            print(f"{path} SKIP")

    # 执行标定并保存结果
    image_sample = cv2.imread(images[0])
    image_size = (image_sample.shape[1], image_sample.shape[0])
    result = calibrate_camera(all_corners, all_ids, board, image_size)
    
    save_result(result, SAVE_PATH)

if __name__ == "__main__":
    main()