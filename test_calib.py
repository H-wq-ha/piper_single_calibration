import numpy as np
import matplotlib.pyplot as plt

# ================= 手眼标定 =================
R = np.array([
    [-0.05835668,  0.919481,   0.38877911],
    [-0.99709039, -0.03455239, -0.06794764],
    [-0.04904331, -0.39161311,  0.91882203]
])

t = np.array([-0.07842231, 0.01022245, 0.03192331])

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

# ================= 1️⃣ 夹爪坐标系（世界系） =================
origin_g = np.array([0, 0, 0])
R_g = np.eye(3)

draw_frame(ax, origin_g, R_g, "G", length=0.07)

# ================= 2️⃣ 相机坐标系 =================
origin_c = t          # 相机位置（在夹爪系下）
R_c = R               # 相机方向（在夹爪系下）

draw_frame(ax, origin_c, R_c, "C", length=0.05)

# ================= 连接线（位移） =================
ax.plot([0, t[0]], [0, t[1]], [0, t[2]],
        'k--', linewidth=1.5, label="translation t")

# ================= 设置 =================
ax.set_xlabel("Xg")
ax.set_ylabel("Yg")
ax.set_zlabel("Zg")

ax.set_title("Camera Pose in Gripper Frame")

# 统一尺度
ax.set_xlim([-0.15, 0.15])
ax.set_ylim([-0.15, 0.15])
ax.set_zlim([-0.15, 0.15])

plt.legend()
plt.show()