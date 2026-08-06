import os

import numpy as np
import matplotlib.pyplot as plt
import yaml

# ================= 加载最新保存的眼在手外标定结果 =================
result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "result", "handeye_eye_to_hand.yaml")

if not os.path.exists(result_path):
    raise FileNotFoundError(f"未找到标定结果文件: {result_path}")

with open(result_path) as f:
    result = yaml.safe_load(f)

R = np.array(result["R_cam2base"])          # 相机姿态（在基座系下）
t = np.array(result["t_cam2base"])          # 相机原点（在基座系下）

# 打印标定结果
print("================ 眼在手外标定结果 (T_cam2base) ================")
print(f"R_cam2base:\n{R}")
print(f"t_cam2base: {t}")

# 提取 ZYX 欧拉角（与手眼标定脚本 euler_to_matrix 约定一致）
def matrix_to_euler(R):
    """旋转矩阵 → ZYX 欧拉角（弧度）"""
    rx = np.arctan2(R[2, 1], R[2, 2])
    ry = np.arcsin(-R[2, 0])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return rx, ry, rz

rx, ry, rz = matrix_to_euler(R)
print(f"欧拉角 (ZYX, 弧度): rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}")
print(f"欧拉角 (ZYX, 角度): rx={np.rad2deg(rx):.2f}, ry={np.rad2deg(ry):.2f}, rz={np.rad2deg(rz):.2f}")

# ================= 画坐标系函数 =================
def draw_frame(ax, origin, R_mat, label, length=0.05):
    colors = ['r', 'g', 'b']
    axes_labels = ['X', 'Y', 'Z']

    for i in range(3):
        axis = R_mat[:, i] * length
        ax.quiver(origin[0], origin[1], origin[2],
                  axis[0], axis[1], axis[2],
                  color=colors[i], linewidth=2)

        ax.text(origin[0] + axis[0],
                origin[1] + axis[1],
                origin[2] + axis[2],
                f"{label}_{axes_labels[i]}",
                color=colors[i])

# ================= 创建图 =================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ================= 1️⃣ 机械臂基座坐标系（世界系） =================
origin_b = np.array([0, 0, 0])
R_b = np.eye(3)

draw_frame(ax, origin_b, R_b, "B", length=0.15)

# ================= 2️⃣ 相机坐标系（在基座系下的位姿） =================
origin_c = t          # 相机位置（在基座系下）
R_c = R               # 相机方向（在基座系下）

draw_frame(ax, origin_c, R_c, "C", length=0.1)

# ================= 连接线（位移） =================
ax.plot([0, t[0]], [0, t[1]], [0, t[2]],
        'k--', linewidth=1.5, label="translation t")

# ================= 设置 =================
ax.set_xlabel("Xb")
ax.set_ylabel("Yb")
ax.set_zlabel("Zb")

ax.set_title("Camera Pose in Base Frame (Eye-to-Hand)")

# 统一尺度（自适应平移量）
span = max(0.2, 1.2 * np.max(np.abs(t)))
ax.set_xlim([-span, span])
ax.set_ylim([-span, span])
ax.set_zlim([-span, span])

plt.legend()
plt.show()
