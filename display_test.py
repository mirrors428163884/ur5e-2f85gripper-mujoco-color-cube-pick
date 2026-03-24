# display_test.py —— 使用隐藏 GLFW 窗口（推荐方式）
import mujoco
import mujoco.viewer
import numpy as np
import time
import glfw
import cv2

# ----------------------------
# 1. 加载模型和数据
# ----------------------------
model = mujoco.MjModel.from_xml_path("/media/sangfor/vdb/robot_code/ur5-mujoco-env/ur5e_robotiq2f85/scene.xml")
data = mujoco.MjData(model)

# ----------------------------
# 2. 相机渲染类（使用隐藏 GLFW 窗口）
# ----------------------------
class HiddenCameraViewer:
    def __init__(self, model, data, camera_name=None, width=640, height=480, is_fixed=True):
        self.model = model
        self.data = data
        self.width, self.height = width, height
        self.is_fixed = is_fixed
        self.name = camera_name or "Camera"

        # 初始化 GLFW（隐藏窗口）
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # 👈 隐藏窗口
        self.window = glfw.create_window(width, height, self.name, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)

        # 相机设置
        self.camera = mujoco.MjvCamera()
        if is_fixed and camera_name:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id == -1:
                raise ValueError(f"Camera '{camera_name}' not found")
            self.camera.fixedcamid = cam_id
        else:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE

        # 渲染资源
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        self.vopt = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()

        # 图像缓冲
        self.rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)

    def update_free_camera(self, lookat, distance=0.6, azimuth=135, elevation=-20):
        if not self.is_fixed:
            self.camera.lookat[:] = lookat
            self.camera.distance = distance
            self.camera.azimuth = azimuth
            self.camera.elevation = elevation

    def render(self):
        # 注意：不需要 make_context_current 如果只有一个 offscreen context
        mujoco.mjv_updateScene(
            self.model, self.data, self.vopt, self.perturb,
            self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb_buffer, None, viewport, self.context)
        rgb_img = np.flipud(self.rgb_buffer)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.name, bgr_img)
        cv2.waitKey(1)

    def close(self):
        glfw.destroy_window(self.window)


# ----------------------------
# 3. 创建两个相机（无可见 GLFW 窗口）
# ----------------------------
cam_world = HiddenCameraViewer(model, data, camera_name="cam_world", is_fixed=True)
cam_follow = HiddenCameraViewer(model, data, camera_name="End-Effector View", is_fixed=False)

# ----------------------------
# 4. 主 passive viewer（唯一可见的 MuJoCo 窗口）
# ----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -30
    viewer.cam.lookat[:] = [0.3, 0.0, 0.4]

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

        # 更新跟随相机
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wrist_3_link')
        if wrist_id != -1:
            wrist_pos = data.xpos[wrist_id]
            cam_follow.update_free_camera(
                lookat=wrist_pos,
                distance=0.5,
                azimuth=135,
                elevation=-20
            )

        # 渲染两个 OpenCV 窗口
        cam_world.render()
        cam_follow.render()

        time.sleep(1 / 60.0)

# ----------------------------
# 5. 清理
# ----------------------------
cam_world.close()
cam_follow.close()
cv2.destroyAllWindows()
glfw.terminate()
