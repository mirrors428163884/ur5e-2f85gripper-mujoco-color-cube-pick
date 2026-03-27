<!-- # ur5-mujoco-env
MuJoCo manipulation sim: UR5 arm with Robotiq 2F85 gripper
### Key Features
- **Pre-configured MJCF Models**:
  - UR5 robotic arm
  - Integrated Robotiq 2F85 adaptive gripper
  - Eye-in-hand camera configuration

- **Core Algorithms**:
  - Forward/Inverse kinematics solver
  - Jacobian matrix computation
  - Cartesian-space linear trajectory planning
  - Gripper state control

- **Demo Scenarios**:
  - Block stacking task
  - Pick-and-place with vision feedback -->


# UR5 MuJoCo 机器人仿真环境

这是一个基于MuJoCo物理引擎的UR5e机械臂仿真项目，集成了Robotiq 2F-85夹爪和视觉系统，用于执行抓取和放置任务。

## 功能特性

- UR5e机械臂与Robotiq 2F-85夹爪的高精度物理仿真
- 多相机视角（世界视角和末端执行器视角）
- 基于颜色的目标检测与定位
- 笛卡尔空间轨迹规划与逆运动学控制
- 视觉伺服抓取任务
- 配置文件驱动的参数化控制

## 安装依赖

### 系统要求
- Ubuntu 20.04 或更高版本
- Python 3.9 或更高版本

### Python 依赖库

```bash
# 基础依赖
pip install mujoco==3.1.6
pip install numpy
pip install opencv-python
pip install PyOpenGL glfw
pip install pyyaml

# 可选依赖（用于YOLO目标检测）
pip install ultralytics

```

### 环境变量设置

```bash
# 设置MuJoCo渲染后端
export MUJOCO_GL=glfw
```

## 项目结构

```
ur5-mujoco-env/
├── ur5e_robotiq2f85/        # UR5e机器人模型文件
│   ├── scene.xml           # 场景定义文件
│   └── ur5e.xml            # UR5e机器人本体定义
├── config.yaml             # 仿真和运动参数配置文件
├── README.md               # 项目说明文档
├── demo.py                 # 基础抓取演示（随机目标）
├── display_test.py         # 相机视图测试
├── input_demo.py           # 交互式抓取演示（用户指定目标）
├── pick_box_env.py         # 基础环境类（支持配置文件）
├── pick_box_env_input.py   # 支持用户输入的环境类
└── vision_pick_main.py     # 视觉伺服抓取主程序
```

## 文件功能说明

### 核心环境类

- **[pick_box_env.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/pick_box_env.py)**: 基础UR5抓取环境类，支持从YAML配置文件加载参数，提供机械臂控制、相机接口和盒子位置管理功能。
- **[pick_box_env_input.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/pick_box_env_input.py)**: 扩展版环境类，支持用户交互式指定抓取和放置目标。

### 演示程序

- **[demo.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/demo.py)**: 自动化抓取演示程序。随机选择一个彩色盒子进行抓取，并放置到另一个随机位置的盒子上。
- **[input_demo.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/input_demo.py)**: 交互式抓取演示程序。用户可以指定要抓取的颜色和放置的目标位置颜色。
- **[vision_pick_main.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/vision_pick_main.py)**: 视觉伺服抓取主程序。使用摄像头实时检测蓝色目标物体，通过视觉反馈进行精确定位和抓取。
- **[display_test.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/display_test.py)**: 相机视图测试程序。展示如何同时显示世界视角和末端执行器视角的相机画面。

### 配置文件

- **[config.yaml](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/config.yaml)**: 包含所有可调参数的配置文件，包括：
  - 运动规划参数（速度、加速度、容差等）
  - 仿真参数（时间步长、控制模式）
  - 盒子参数（颜色映射、随机生成范围）
  - 抓取参数（接近高度、抓取高度、抬升高度等）

## 使用方法

### 1. 基础抓取演示
```bash
python demo.py
```

### 2. 交互式抓取演示
```bash
python input_demo.py
```
程序会提示您输入要抓取和放置的颜色（red/green/blue）。

### 3. 视觉伺服抓取
```bash
python vision_pick_main.py
```
程序会自动检测场景中的蓝色物体并执行抓取任务。

### 4. 相机视图测试
```bash
python display_test.py
```
查看多相机视角的渲染效果。

## 参数配置

所有运动和仿真参数都可以通过编辑 [config.yaml](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/config.yaml) 文件进行调整，无需修改代码。

### 运动参数调整
- `cartesian_velocity`: 笛卡尔空间移动速度
- `cartesian_acceleration`: 笛卡尔空间加速度
- `max_joint_step`: 单次迭代的最大关节角度变化
- `tolerance`: 位置到达容差

### 仿真参数调整
- [dt](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/pick_box_env.py#L0-L0): 仿真时间步长
- `sleep_control`: 是否启用时间同步控制

### 抓取参数调整
- `approach_height`: 接近目标时的安全高度
- `grasp_height`: 实际抓取高度
- `lift_height`: 抓取后的抬升高度
- `place_height`: 放置时的高度

## 注意事项

1. **MuJoCo许可证**: 需要有效的MuJoCo许可证才能运行仿真。
2. **性能要求**: 实时视觉处理对CPU有一定要求，建议在性能较好的机器上运行。
3. **相机标定**: [vision_pick_main.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/vision_pick_main.py) 中的 [VISION_OFFSET](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/vision_pick_main.py#L17-L17) 参数需要根据实际手眼标定结果进行调整。
4. **目标检测**: 视觉伺服版本默认使用颜色检测，如需使用YOLO等深度学习方法，请确保安装了 `ultralytics` 库。

## 扩展开发

- 修改 [scene.xml](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/ur5e_robotiq2f85/scene.xml) 添加新的物体或改变场景布局
- 调整 [config.yaml](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/config.yaml) 中的参数优化运动性能
- 在 [vision_pick_main.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/vision_pick_main.py) 中扩展目标检测算法
- 基于现有环境类开发新的任务逻辑

## 故障排除

**问题**: MuJoCo初始化失败
- **解决方案**: 确保已正确设置MUJOCO_GL环境变量和许可证

**问题**: 相机画面不显示
- **解决方案**: 确保已安装OpenCV (`pip install opencv-python`)

**问题**: 机械臂运动异常
- **解决方案**: 检查 [config.yaml](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/config.yaml) 中的运动参数是否合理，特别是速度和加速度值

**问题**: 目标检测失败
- **解决方案**: 调整 [vision_pick_main.py](file:///media/sangfor/vdb/robot_code/ur5-mujoco-env/vision_pick_main.py) 中的颜色检测阈值或检查光照条件
```
