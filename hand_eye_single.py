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
SQUARE_LENGTH = 0.013     # 方格边长（米）
MARKER_LENGTH = 0.009     # ArUco 标记边长（米）
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
# 6. 手眼标定器（原 handeye/calibrator.py）
# ============================================================
class HandEyeCalibrator:
    @staticmethod
    def _solve_ax_xb(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
        """
        求解 AX=XB 手眼标定（numpy 实现，替代 OpenCV 5.0 移除的 cv2.calibrateHandEye）

        相邻位姿对构造相对运动（同为 i → i+1 方向）：
            A_i = inv(T_gripper2base_i) @ T_gripper2base_{i+1}
            B_i = T_target2cam_i @ inv(T_target2cam_{i+1})

        旋转部分：R_A @ R_X = R_X @ R_B 对 vec(R_X) 是线性的，
        堆叠所有位姿对得 (I⊗R_A − R_Bᵀ⊗I) vec(R_X) = 0，SVD 求零空间后正交化到 SO(3)。
        平移部分：(R_A − I) @ t_X = R_X @ t_B − t_A，最小二乘。

        返回 (R_X, t_X)：
            眼在手内传入 T_gripper2base → X = T_cam2gripper
            眼在手外传入 T_base2gripper → X = T_cam2base
        """
        n = len(R_gripper2base)
        if n < 3:
            raise ValueError("至少需要 3 组数据才能构造相对运动")

        A_R, A_t, B_R, B_t = [], [], [], []
        for i in range(n - 1):
            R_i, t_i = R_gripper2base[i], t_gripper2base[i].reshape(3)
            R_j, t_j = R_gripper2base[i + 1], t_gripper2base[i + 1].reshape(3)
            A_R.append(R_i.T @ R_j)
            A_t.append(R_i.T @ (t_j - t_i))
            B_R.append(R_target2cam[i] @ R_target2cam[i + 1].T)
            B_t.append(t_target2cam[i].reshape(3)
                       - R_target2cam[i] @ R_target2cam[i + 1].T @ t_target2cam[i + 1].reshape(3))

        # 旋转：堆叠线性方程，SVD 最小奇异值向量即 vec(R_X)
        M = np.zeros((9 * (n - 1), 9))
        for k in range(n - 1):
            M[9*k:9*k+9] = np.kron(np.eye(3), A_R[k]) - np.kron(B_R[k].T, np.eye(3))
        _, _, Vh = np.linalg.svd(M)
        # vec 为列主序，必须按列填充还原矩阵
        R_x = Vh[-1].reshape(3, 3, order='F')

        # 正交化；Vh[-1] 的符号是任意的，det<0 说明取了反号，整体取反
        U, _, Vt = np.linalg.svd(R_x)
        R_x = U @ Vt
        if np.linalg.det(R_x) < 0:
            R_x = -R_x

        # 平移：最小二乘
        P = np.zeros((3 * (n - 1), 3))
        q = np.zeros(3 * (n - 1))
        for k in range(n - 1):
            P[3*k:3*k+3] = A_R[k] - np.eye(3)
            q[3*k:3*k+3] = R_x @ B_t[k] - A_t[k]
        t_x, *_ = np.linalg.lstsq(P, q, rcond=None)

        return R_x, t_x

    def calibrate(self, collector):
        """
        执行手眼标定（AX=XB 线性最小二乘）

        参数：
            collector: HandEyeDataCollector 实例

        返回：
            R_cam2gripper (3x3), t_cam2gripper (3,)
            即相机坐标系到机械臂末端坐标系的变换
        """
        return self._solve_ax_xb(
            collector.R_gripper2base,
            collector.t_gripper2base,
            collector.R_target2cam,
            collector.t_target2cam,
        )


# ============================================================
# 7. 机械臂接口
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


def save_handeye_result(R, t, path="data/result/handeye.yaml"):
    """保存手眼标定结果到 YAML"""
    result = np.eye(4)
    result[:3, :3] = R
    result[:3, 3] = t.flatten()

    data = {
        "R_cam2gripper": R.tolist(),
        "t_cam2gripper": t.tolist(),
        "T_cam2gripper": result.tolist(),
    }
    with open(path, "w") as f:
        yaml.dump(data, f)
    print(f"标定结果已保存至: {path}")


# ============================================================
# 9. 主流程（原 main_handeye.py）
# ============================================================
def main():
    # ===== 初始化所有组件 =====
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

    print("\n========== 操作说明 ==========")
    print("  S  — 采样（采集视觉位姿 + 机械臂位姿）")
    print("  C  — 执行手眼标定（需要 >= 10 组数据）")
    print("  Q  — 退出")
    print("===============================\n")

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

            cv2.imshow("HandEye - Single File", display)

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

                print("开始手眼标定...")
                R, t = calibrator.calibrate(collector)

                print("\n===== 手眼标定结果 =====")
                print("R_cam2gripper:\n", R)
                print("t_cam2gripper:\n", t)

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
