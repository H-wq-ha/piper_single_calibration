# 开发日志 — piper_single_calibration

> 记录时间:2026-08-06
> 仅记录实际完成的工作

---

## 1. 项目概述

松灵 Piper 机械臂手眼标定项目(单文件实现),RealSense 相机 + ChArUco 标定板(9×12,方格 13mm / 标定码 9mm,DICT_5X5_100),支持眼在手上 / 眼在手外两种布置。

| 文件 | 作用 |
|---|---|
| `intrinsics/main_capture.py` | 相机内参标定(采集 + 自动标定) |
| `hand_eye_single.py` | 眼在手上手眼标定 |
| `hand_eye_single_eye_to_hand.py` | 眼在手外手眼标定 |
| `test_calib.py` | 标定结果可视化(硬编码示例数据) |
| `data/result/calibration.yaml` | 相机内参结果 |

---

## 2. 环境搭建(已完成)

创建 `venv` 虚拟环境并安装最新依赖:

```bash
python3 -m venv venv
venv/bin/pip install opencv-python pyrealsense2 numpy pyyaml matplotlib scipy piper_sdk
```

| 包 | 版本 |
|---|---|
| opencv-python | **5.0.0** |
| numpy | 2.2.6 |
| pyrealsense2 | 2.58.3 |
| piper_sdk | 0.6.2 |
| matplotlib / scipy | 3.10.9 / 1.15.3 |

实测验证过的兼容性事实:
- numpy 2.2.6 与系统包 matplotlib 3.5.1 不兼容(导入即崩溃)、scipy 1.11.4 报警告 → 必须使用 venv
- OpenCV 4.11 与 5.0 均已移除旧版 aruco API(`detectMarkers` / `interpolateCornersCharuco` / `calibrateCameraCharuco`)

---

## 3. OpenCV 5.0 适配改造(已完成)

### 3.1 重写 `intrinsics/main_capture.py`(新 aruco API)

- `CharucoDetector.detectBoard` 替代 `detectMarkers` + `interpolateCornersCharuco`
- `board.matchImagePoints` + `cv2.calibrateCamera` 替代 `calibrateCameraCharuco`
- 新增 `--calibrate-only` 模式:跳过采集,直接用 `data/images/*.png` 标定(见第 5 节)

两个手眼脚本的 aruco 部分本就使用新 API(`CharucoDetector` + `matchImagePoints`),无需改动。

### 3.2 numpy 实现 AX=XB 手眼标定求解器(替换 `cv2.calibrateHandEye`)

OpenCV 5.0 删除了 `cv2.calibrateHandEye`(仅保留 `CALIB_HAND_EYE_*` 常量),在两个手眼脚本的 `HandEyeCalibrator` 中实现了等价求解器:

**算法**:相邻位姿对构造相对运动 `A_i = inv(T_g2b_i)·T_g2b_{i+1}`、`B_i = T_t2c_i·inv(T_t2c_{i+1})`,解 `AX = XB`:

- 旋转:`R_A·R_X = R_X·R_B` 对 `vec(R_X)` 线性,堆叠成 `(I⊗R_A − R_Bᵀ⊗I)·vec(R_X) = 0`,SVD 取最小奇异值向量,再正交化到 SO(3)
- 平移:`(R_A − I)·t_X = R_X·t_B − t_A`,最小二乘
- 眼在手内传入 `T_gripper2base` 得 `X = T_cam2gripper`;眼在手外传入 `T_base2gripper`(取逆)得 `X = T_cam2base`

**调试中修复的 3 个实现 bug**:

| # | 问题 | 现象 | 修复 |
|---|---|---|---|
| 1 | A、B 相对运动方向混搭 | R 误差 ~150° | A、B 统一为 i→i+1 方向 |
| 2 | `vec` 为列主序但 `reshape` 按行填充 | 结果等于转置 | `reshape(3, 3, order='F')` |
| 3 | `Vh[-1]` 符号任意 | 噪声下 R 误差恰为 180° | SVD 正交化后 `det<0` 时整体取反(而非翻转单列) |

**验证结果**:

| 测试 | R 误差 | t 误差 |
|---|---|---|
| 合成真值 ×20 种子(眼在手上) | 1.7e-06° | 5e-16 m |
| 合成真值 ×20 种子(眼在手外) | 1.2e-06° | 4e-16 m |
| 加噪声(0.5mm + 0.2° 测量噪声) | 0.11° | 0.52 mm |

---

## 4. 实机验证(已完成)

### 4.1 定位并修复 `drawDetectedCornersCharuco` 断言错误(OpenCV 5.0 bug)

**现象**:检测到标定板即报错:

```
error: (-215:Assertion failed) (_charucoCorners.total() == _charucoIds.total()) || _charucoIds.total() == 0
```

**根因**:OpenCV 5.0 将 charuco 角点形状从 4.x 的 `(N,1,2)` 改为 `(N,2)`,而 `drawDetectedCornersCharuco` 内部仍用 `total()` 比较:`(N,2)` 的 `total() = 2N` 永远不等于 `ids.total() = N`。实测确认角点/ids 数量本身无不一致(250 帧零不匹配)。

**修复**:弃用该函数,手工 `cv2.circle` 画角点;`detect_charuco` 增加防御性截断(`min(len(corners), len(ids))`)。

### 4.2 相机检验(RealSense D435I)

| 项目 | 结果 |
|---|---|
| 彩色 / 深度流 | 640×480 @ 30.0 fps 稳定 |
| 标定板检测(板距 29cm) | 250/250 帧,**88/88 角点**完整检出 |
| 位姿估计 | 250/250 成功,板距 29±0 cm |
| 设备内参 vs 标定内参 | fx 604.2 vs 604.1(一致) |

关键结论:检测率取决于板距相机距离——约 1m 时 marker 仅 ~8px,检出率极低;约 30cm 时 marker ~28px,检测率 100%。

### 4.3 机械臂控制测试(piper_sdk)

- CAN 通信正常(can0,帧率 ~3270/s),平移运动可靠(到位误差 <1mm)
- 观察到 RZ 旋转在 RZ≈93° 附近失败(疑似腕部关节限位),RX/RY 部分方向可用

---

## 5. 内参标定(已完成)

- 新增 `--calibrate-only` 模式:原流程自动标定步骤绑定在"按 Q 退出后",Ctrl+C 或关窗会导致标定不执行且无提示;该模式可直接用 `data/images/` 下已保存的照片标定,无需相机
- 使用 `data/images/` 下 41 张已保存照片执行标定:

```bash
venv/bin/python intrinsics/main_capture.py --calibrate-only
```

**结果**(已保存至 `data/result/calibration.yaml`):

| 项目 | 数值 |
|---|---|
| K | fx=604.1, fy=601.1, cx=340.8, cy=238.7 |
| 畸变 | [0.083, 0.053, -0.007, 0.010, -0.555] |
| 重投影误差 | **0.42 px** |

---

## 6. 常用命令

```bash
# 内参标定(采集 + 自动标定)
venv/bin/python intrinsics/main_capture.py
# 内参标定(仅用已存照片)
venv/bin/python intrinsics/main_capture.py --calibrate-only
# 眼在手上标定
venv/bin/python hand_eye_single.py
# 眼在手外标定
venv/bin/python hand_eye_single_eye_to_hand.py
# 结果可视化
venv/bin/python test_calib.py
```
