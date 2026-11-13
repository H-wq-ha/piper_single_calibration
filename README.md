# piper_single_calibration

松灵 Piper 机械臂＋RealSense 相机手眼标定，支持眼在手外和眼在手上。手眼求解使用 OpenCV 4.12 官方实现，默认采用 Shah 联合标定。

关键修改记录见 [开发日志](docs/DEVELOPMENT_LOG.md)。

## 1. 安装

需要 Python 3.10 或更新版本，在项目根目录执行：

```bash
python3 -m venv venv
# 已有环境先移除其他提供 cv2 的发行包，避免冲突
venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
venv/bin/python -m pip install -r requirements.txt
# 连接 RealSense 或 Piper 时安装硬件依赖
venv/bin/python -m pip install -r requirements-hardware.txt
```

依赖固定为 `opencv-contrib-python==4.12.0.88`、NumPy 2.2.6、SciPy 1.15.3。当前求解代码需要 OpenCV 4.12 的手眼接口。

## 2. 相机内参标定

使用项目附带的 Calib.io ChArUco PDF：8×11 方格，按 PDF 横向定义为 **11 列×8 行**，方格边长 **18 mm**，标记边长 **13 mm**。采用 `DICT_5X5_100` 和 `setLegacyPattern(True)`，与该文件的 marker 布局匹配。内参与手眼程序共用 `handeye/board.py` 中的板定义。打印后核对实际尺寸和平整度。相机彩色流为 640×480、30 fps。

```bash
venv/bin/python intrinsics/main_capture.py
# 仅使用最新一批已保存的图像重新标定
venv/bin/python intrinsics/main_capture.py --calibrate-only
# 指定某批图像（也支持历史 data/images 目录）
venv/bin/python intrinsics/main_capture.py --calibrate-only --images-dir data/images/<批次>
# 如需保留空格保存的操作方式
venv/bin/python intrinsics/main_capture.py --manual
```

默认自动保存：至少检测到 20 个角点、角点区域占画面至少 4%、板内清晰度通过检查，并停稳 0.6 秒后保存。自动保存至少间隔 1.5 秒；与已保存照片视角相近时提示移动或倾斜标定板。建议采集 20–40 张，覆盖中心、边缘、不同距离和两个方向的倾角。清晰度是采集筛选指标，不代表已完成精度验收；可通过 `--min-sharpness` 调整，其他参数见 `--help`。

按 **A** 暂停/恢复自动保存，按 **空格** 手动保存质量合格的当前帧（跳过静止、间隔和重复视角限制），按 **Q / Esc** 或关闭窗口结束采集并标定。至少需要 5 张有效照片；不足时保留照片，不修改原内参。

每次采集保存到独立的 `data/images/<时间戳>/`，包括不带叠加标记的原始 PNG、板/采集配置 `session.json` 和角点/质量记录 `samples.jsonl`；会话记录相机流内参，不保存设备序列号。**每次启动采集并成功打开相机和窗口后，自动删除 `data/images/` 中先前批次的 PNG 照片，仅保留本次照片；旧批次元数据及已生成内参和备份保留。** 如果需要保留照片，请在下次采集前复制到 `data/images/` 之外。`--calibrate-only` 不清理照片；程序通过互斥锁避免另一轮内参程序使用照片时执行清理。

内参写入 `data/result/calibration.yaml`，覆盖前自动创建带时间戳的备份；结果增加板定义、分辨率、源照片路径、逐帧 RMS 和内参标准差，便于追溯。内参计算仍使用 OpenCV 标准五参数畸变模型，未实现独立物理精度验收。

## 3. 手眼数据采集

先配置 CAN 接口（默认 `can0`、1 Mbps；脚本需要系统安装 `ethtool` 和 `can-utils`）：

```bash
bash can_activate.sh can0 1000000
```

根据安装方式选择入口：

```bash
# 眼在手外：相机固定，标定板刚性安装在末端
venv/bin/python hand_eye_single_eye_to_hand.py
# 眼在手上：相机安装在末端，标定板固定在场景
venv/bin/python hand_eye_single.py
```

统一入口支持指定内参、CAN 接口和保存目录：

```bash
venv/bin/python -m handeye capture --mode eye_to_hand --can-port can0 --intrinsics data/result/calibration.yaml --output data/sessions
# 眼在手上时使用 --mode eye_in_hand
```

手动移动机械臂，等待静止后保存；姿态应覆盖不同旋转轴、距离和图像区域。

| 按键 | 用途 |
|---|---|
| S | 保存训练帧 |
| V | 保存开发帧，用于显式 auto 选方法 |
| T | 保存独立测试帧 |
| Q | 结束采集 |

程序检查角点、PnP 质量、平面歧义、机器人反馈新鲜度及静止状态。默认至少 20 个角点、视觉 RMSE ≤0.8 px；静止窗口 0.5 s，位移 ≤0.5 mm、旋转 ≤0.15°。采集入口不发送运动指令；如需使能机械臂，显式添加 `--enable-arm`。

每批数据保存到唯一的 `data/sessions/<批次>/`，包含原始 PNG、`session.json`、`samples.jsonl`、相机流内参及拒绝记录，不保存设备序列号。有效样本记录机器人位姿、板位姿、角点与 ID、时间戳和图像 SHA256；载入时校验图像。

## 4. 离线求解

```bash
venv/bin/python -m handeye solve data/sessions/<批次> --output output/<新结果目录>
# 显式切换方法
venv/bin/python -m handeye solve data/sessions/<批次> --output output/<另一个新目录> --method opencv_park
# 未采集完整 S/V/T 分组时，显式采用连续 60/20/20 划分（至少 10 帧）
venv/bin/python -m handeye solve data/sessions/<批次> --output output/<新目录> --auto-split
```

输出目录必须不存在。可选方法为 `opencv_shah`（默认）、`opencv_li`、`opencv_tsai`、`opencv_park`、`opencv_horaud`、`opencv_andreff`、`opencv_daniilidis`。仅显式指定 `--method auto` 时，根据开发集链路像素 RMSE 从官方方法中选取；固定方法失败会报错。

| 输出文件 | 内容 |
|---|---|
| `handeye.yaml` | 可直接读取的外参 R、t、T，以及标定板固定变换 |
| `calibration.json` | 模式、X/Y、方法、单位和后端信息 |
| `evaluation.json` | 训练/开发/测试划分、各官方方法统计及失败原因 |
| `per_frame.csv` | 每帧链路像素误差和位姿一致性误差 |

求解结果写入指定目录，不自动覆盖 `data/result/` 中已有的物理标定参数。新运行结果目录 `output/` 已加入忽略规则。

## 5. 坐标与结果使用

矩阵采用列向量左乘，`T_a2b` 将 a 系坐标转换到 b 系，平移单位为米。

| 模式 | X / YAML 字段 | Y / YAML 字段 |
|---|---|---|
| 眼在手外 | 相机到基座 / `T_cam2base` | 标定板到末端 / `T_target2gripper` |
| 眼在手上 | 相机到末端 / `T_cam2gripper` | 标定板到基座 / `T_target2base` |

令 G 为末端到基座、C 为标定板到相机，则 `C = inv(X) @ S @ Y`；眼在手外 `S=G`，眼在手上 `S=inv(G)`。眼在手外的点转换为 `p_base = X @ p_camera`；眼在手上为 `p_base = G @ X @ p_camera`，点使用齐次坐标。

像素误差与相对于 PnP 的毫米/角度一致性用于检查标定链路。实际定位或抓取精度需要另做独立 TCP 对点和工况验证；导出结果的 `physical_acceptance` 默认是 `false`。

## 6. 保留的核心文件

- `intrinsics/main_capture.py`：相机内参采集与标定。
- `hand_eye_single.py`、`hand_eye_single_eye_to_hand.py`：两种安装方式的采集入口。
- `handeye/`：采集、存储、视觉前端、坐标变换、运动诊断、官方求解和结果导出。
- `can_activate.sh`、`requirements*.txt`、标定板 PDF：硬件配置、依赖和打印板。
- `data/`：已有图像、原始数据、会话和标定参数，运行数据默认忽略提交。

旧测试/公开数据评测入口、研究绘图、旧算法快照和历史报告已清理；使用 `python -m handeye --help` 查看当前命令。
