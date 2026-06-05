import cv2
import yaml
import time
import numpy as np
import pyrealsense2 as rs

# ============================================================
# 1. 配置参数（原 config.py）
# ============================================================
SQUARES_X = 12            # ChArUco 棋盘列数
SQUARES_Y = 9             # ChArUco 棋盘行数
SQUARE_LENGTH = 0.017     # 方格边长（米）
MARKER_LENGTH = 0.012     # ArUco 标记边长（米）
ARUCO_DICT = "DICT_5X5_100"
MIN_CORNERS = 20          # 最小有效角点数（过滤低质量帧）
CALIBRATION_PATH = "data/result/calibration.yaml"  # 相机内参路径


# ============================================================
# 2. 标定板创建（原 board/charuco_board.py）
# ============================================================
DICT_MAP = {
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}


def create_charuco_board():
    """创建 ChArUco 标定板对象和字典"""
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP[ARUCO_DICT])
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )
    return board, dictionary


# ============================================================
# 3. 角点检测器（原 detection/charuco_detector.py）
# ============================================================
class CharucoDetector:
    def __init__(self, board, dictionary):
        self.board = board
        self.dictionary = dictionary
        # OpenCV 4.10+ CharucoDetector 合并了 ArUco 检测和 ChArUco 插值
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    def detect(self, image):
        """检测 ChArUco 角点，返回 (charuco_corners, charuco_ids)"""
        charuco_corners, charuco_ids, _, _ = self.detector.detectBoard(image)
        if charuco_corners is None or len(charuco_corners) == 0:
            return None, None
        return charuco_corners, charuco_ids


# ============================================================
# 4. 位姿估计器（原 pose_estimation/charuco_pose.py）
# ============================================================
class CharucoPoseEstimator:
    def __init__(self, board, K, dist):
        self.board = board
        self.K = K
        self.dist = dist

    def estimate_pose(self, corners, ids):
        """估计标定板在相机坐标系下的位姿，返回 (ok, rvec, tvec)"""
        if corners is None or ids is None:
            return False, None, None

        # OpenCV 4.10+ 用 matchImagePoints + solvePnP 替代 estimatePoseCharucoBoard
        obj_points, img_points = self.board.matchImagePoints(corners, ids)
        if obj_points is None or len(obj_points) < 6:
            return False, None, None

        valid, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not valid:
            return False, None, None
        return True, rvec, tvec

    @staticmethod
    def rvec_to_matrix(rvec):
        """旋转向量 → 旋转矩阵"""
        R, _ = cv2.Rodrigues(rvec)
        return R

    @staticmethod
    def build_transform(rvec, tvec):
        """构建 4x4 齐次变换矩阵 T_target2cam"""
        R = CharucoPoseEstimator.rvec_to_matrix(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T


# ============================================================
# 5. 数据收集器（原 handeye/data_collector.py）
# ============================================================
class HandEyeDataCollector:
    """
    眼在手外数据收集器

    眼在外场景：相机固定，标定板装在机械臂末端。
    每次采集团组数据：
      - T_target2cam: 标定板在相机坐标系下的位姿（视觉检测）
      - T_gripper2base: 机械臂末端在基座坐标系下的位姿（机器人反馈）
    """
    def __init__(self):
        self.R_target2cam = []
        self.t_target2cam = []
        self.R_gripper2base = []
        self.t_gripper2base = []

    def add_sample(self, T_target2cam, T_gripper2base):
        """添加一组配对数据（两个 4x4 齐次矩阵）"""
        self.R_target2cam.append(T_target2cam[:3, :3])
        self.t_target2cam.append(T_target2cam[:3, 3])
        self.R_gripper2base.append(T_gripper2base[:3, :3])
        self.t_gripper2base.append(T_gripper2base[:3, 3])

    @property
    def count(self):
        return len(self.R_target2cam)

    def is_ready(self):
        """是否满足标定所需最少数据量（>=10 组）"""
        return self.count >= 10


# ============================================================
# 6. 手眼标定器（眼在手外版本）
# ============================================================
class HandEyeCalibrator:
    """
    眼在手外（Eye-to-Hand）标定器

    眼在手外场景：
      相机固定于外部（三脚架等），标定板固定于机械臂末端。
      不同位姿下，标定板在相机中的位姿变化，结合机械臂末端位姿变化，
      解算相机坐标系到机械臂基座坐标系的变换 T_cam2base。

    OpenCV calibrateHandEye 对于眼在手外的用法：
      输入 A = T_base2gripper（即 T_gripper2base 的逆）
      输入 B = T_target2cam（标定板在相机中的位姿）
      输出 X = T_cam2base（相机在基座坐标系下的位姿）


      Base → End → Board
      Base → Camera → Board
      基座到夹爪变换 * 夹爪到标版变换（固定） = 基座到相机变换（未知） * 相机到标版变换
    """

    @staticmethod
    def invert_transform(R, t):
        """求 4x4 齐次变换矩阵的逆：返回 (R_inv, t_inv)"""
        R_inv = R.T
        t_inv = (-R_inv @ t.reshape(3, 1)).flatten()
        return R_inv, t_inv

    def calibrate(self, collector):
        """
        执行眼在手外标定（Tsai 方法）

        参数：
            collector: HandEyeDataCollector 实例

        返回：
            R_cam2base (3x3), t_cam2base (3,)
            即相机坐标系到机械臂基座坐标系的变换
        """
        # 将 T_gripper2base 求逆，得到 T_base2gripper
        R_base2gripper = []
        t_base2gripper = []
        for R_g2b, t_g2b in zip(collector.R_gripper2base, collector.t_gripper2base):
            R_b2g, t_b2g = self.invert_transform(R_g2b, t_g2b)
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)

        # 眼在手外：输入 T_base2gripper 和 T_target2cam，输出 T_cam2base
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_base2gripper,
            t_base2gripper,
            collector.R_target2cam,
            collector.t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        return R_cam2base, t_cam2base


# ============================================================
# 7. 机械臂接口（原 handeye/robot_interface.py）
# ============================================================
from piper_sdk import *


class RobotInterface:
    def __init__(self):
        self.piper = C_PiperInterface_V2()

    def enable(self):
        """连接并使能机械臂"""
        self.piper.ConnectPort()
        time.sleep(0.1)
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        print("使能成功!!!!")

    @staticmethod
    def euler_to_matrix(rx, ry, rz):
        """欧拉角 → 旋转矩阵（ZYX 顺序，即 Intrinsic RZ -> RY -> RX）"""
        cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
        sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)

        Rz = np.array([
            [cz, -sz, 0],
            [sz,  cz, 0],
            [0,   0,  1]
        ])
        Ry = np.array([
            [cy, 0,  sy],
            [0,  1,  0],
            [-sy, 0, cy]
        ])
        Rx = np.array([
            [1, 0,  0],
            [0, cx, -sx],
            [0, sx,  cx]
        ])
        return Rz @ Ry @ Rx

    def get_pose(self):
        """
        获取当前机械臂末端在基座坐标系下的位姿

        返回：
            T_gripper2base (4x4 齐次变换矩阵)
        """
        gripper = self.piper.GetArmEndPoseMsgs().end_pose

        # 单位转换：微米 → 米，毫度 → 弧度
        x = gripper.X_axis * 1e-6
        y = gripper.Y_axis * 1e-6
        z = gripper.Z_axis * 1e-6

        rx = np.deg2rad(gripper.RX_axis * 1e-3)
        ry = np.deg2rad(gripper.RY_axis * 1e-3)
        rz = np.deg2rad(gripper.RZ_axis * 1e-3)

        R = self.euler_to_matrix(rx, ry, rz)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        print(f"T_gripper2base:\n{T}")
        return T


# ============================================================
# 8. 工具函数（原 utils/io_utils.py）
# ============================================================
def load_calibration(path):
    """从 YAML 文件加载相机内参"""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeff"])
    return K, dist


def save_handeye_result(R, t, path="data/result/handeye_eye_to_hand.yaml"):
    """保存眼在手外标定结果到 YAML（T_cam2base）"""
    result = np.eye(4)
    result[:3, :3] = R
    result[:3, 3] = t.flatten()

    data = {
        "description": "Eye-to-Hand calibration result: T_cam2base",
        "R_cam2base": R.tolist(),
        "t_cam2base": t.tolist(),
        "T_cam2base": result.tolist(),
    }
    with open(path, "w") as f:
        yaml.dump(data, f)
    print(f"标定结果已保存至: {path}")


# ============================================================
# 9. 主流程（眼在手外标定）
# ============================================================
def main():
    # ===== 初始化所有组件 =====
    print("========== 眼在手外（Eye-to-Hand）标定 ==========")
    print("正在初始化...")
    board, dictionary = create_charuco_board()
    K, dist = load_calibration(CALIBRATION_PATH)
    print(f"相机内参已加载: {CALIBRATION_PATH}")

    detector = CharucoDetector(board, dictionary)
    pose_estimator = CharucoPoseEstimator(board, K, dist)
    collector = HandEyeDataCollector()
    calibrator = HandEyeCalibrator()
    robot = RobotInterface()

    # ===== 启动 RealSense 相机 =====
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_config)

    # ===== 启动机械臂 =====
    robot.enable()

    print("\n========== 操作说明（眼在手外）==========")
    print("  物理设置：相机固定在外部（三脚架等），标定板装在机械臂末端")
    print("  S  — 采样（采集视觉位姿 + 机械臂位姿）")
    print("  C  — 执行手眼标定（需要 >= 10 组数据）")
    print("  Q  — 退出")
    print("  标定结果：T_cam2base（相机坐标系 → 基座坐标系）")
    print("============================================\n")

    try:
        while True:
            # ===== 获取相机帧 =====
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            display = img.copy()

            # ===== 检测并估计位姿 =====
            corners, ids = detector.detect(img)
            ok, rvec, tvec = pose_estimator.estimate_pose(corners, ids)

            # ===== 可视化 =====
            if ok:
                cv2.drawFrameAxes(display, K, dist, rvec, tvec, 0.05)
                status = "POSE OK"
                color = (0, 200, 0)
            else:
                status = "NO POSE"
                color = (0, 0, 255)

            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"Samples: {collector.count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, "Eye-to-Hand",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            cv2.imshow("HandEye - Eye-to-Hand", display)

            # ===== 键盘响应 =====
            key = cv2.waitKey(1)

            if key == ord('s'):
                if not ok:
                    print("无有效视觉位姿，跳过采样")
                    continue

                print("正在获取机械臂位姿...")
                T_gripper2base = robot.get_pose()
                T_target2cam = CharucoPoseEstimator.build_transform(rvec, tvec)

                collector.add_sample(T_target2cam, T_gripper2base)
                print(f"已采样: {collector.count} 组\n")

            elif key == ord('c'):
                if not collector.is_ready():
                    print(f"数据不足（当前 {collector.count} 组，需要 >= 10 组）")
                    continue

                print("开始眼在手外标定...")
                R, t = calibrator.calibrate(collector)

                print("\n===== 眼在手外标定结果 =====")
                print("R_cam2base:\n", R)
                print("t_cam2base:\n", t)

                # 计算并显示 T_cam2base 完整矩阵
                T_cam2base = np.eye(4)
                T_cam2base[:3, :3] = R
                T_cam2base[:3, 3] = t.flatten()
                print("T_cam2base (4x4):\n", T_cam2base)

                # 保存结果
                save_handeye_result(R, t)
                print("")

            elif key == ord('q'):
                print("退出")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
