# piper_single_calibration

使用单文件实现对松灵 piper 机械臂的手眼标定（RealSense 相机 + ChArUco 标定板）。

## 环境要求

- Python 3.10+
- 兼容 **OpenCV 4.10+ 与 5.x**（代码仅使用新版 aruco API：`CharucoDetector` / `matchImagePoints`；
  `cv2.calibrateHandEye` 与旧版 aruco API 已在 OpenCV 5.0 移除，本项目用 numpy 实现了等价的 AX=XB 求解器）
- RealSense 相机、Piper 机械臂（`piper_sdk`）

```bash
python3 -m venv .venv
.venv/bin/pip install opencv-python pyrealsense2 numpy pyyaml matplotlib scipy piper_sdk
```

## 使用步骤

1. **相机内参标定**（首次必做；标定板尺寸已按方格 13mm / 标定码 9mm 配置）：

   ```bash
   .venv/bin/python intrinsics/main_capture.py
   ```

   画面显示 GOOD 时按**空格**保存（建议 10~20 张，覆盖不同角度/距离），按 **Q** 退出后自动标定，
   结果保存至 `data/result/calibration.yaml`。

2. **手眼标定**（二选一，根据相机安装方式）：

   ```bash
   .venv/bin/python hand_eye_single.py              # 眼在手上：相机装在机械臂末端
   .venv/bin/python hand_eye_single_eye_to_hand.py  # 眼在手外：相机固定，标定板装在末端
   ```

   - 程序自动连接 RealSense 与机械臂，画面显示 `POSE OK` 时按 **S** 采样
   - 每换一个姿态按一次 S（**至少 10 组**，姿态要有明显旋转差异，否则标定退化）
   - 采完后按 **C** 标定并保存：
     - 眼在手上 → `data/result/handeye.yaml`（`T_cam2gripper`）
     - 眼在手外 → `data/result/handeye_eye_to_hand.yaml`（`T_cam2base`）
   - 按 **Q** 退出

3. **结果可视化**：

   ```bash
   .venv/bin/python test_calib.py
   ```

## 注意事项

- 标定板实际边长以打印件为准（PDF 文件名中的 13_9 对应方格 13mm / 标定码 9mm），
  尺寸配置在三个脚本顶部，换板后需同步修改。
- 采样姿态多样性决定精度：保证末端旋转轴方向有变化，避免纯平移或绕同一轴旋转。
